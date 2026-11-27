# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

@tool
extends EditorPlugin

const INTERVAL := 1.0 / 15.0
var endpoint := ""
var secret := ""
var instance_id := ""
var active := false
var input_busy := false
var frame_busy := false
var elapsed := 0.0
var input_request: HTTPRequest
var frame_request: HTTPRequest
var dock: VBoxContainer
var pairing: LineEdit
var status: Label
var last_camera: Camera3D
var last_transform := Transform3D.IDENTITY
var pivot := Vector3.ZERO

func _enter_tree() -> void:
	input_request = HTTPRequest.new()
	frame_request = HTTPRequest.new()
	for request in [input_request, frame_request]:
		request.timeout = 0.75
		request.max_redirects = 0
		add_child(request)
	input_request.request_completed.connect(_receive_input)
	frame_request.request_completed.connect(_receive_frame)
	dock = VBoxContainer.new()
	dock.name = "Unsloth Preview"
	pairing = LineEdit.new()
	pairing.placeholder_text = "Pairing code from Unsloth Studio"
	pairing.secret = true
	dock.add_child(pairing)
	var connect_button := Button.new()
	connect_button.text = "Connect preview"
	connect_button.pressed.connect(func(): connect_preview(pairing.text))
	dock.add_child(connect_button)
	var disconnect_button := Button.new()
	disconnect_button.text = "Disconnect"
	disconnect_button.pressed.connect(disconnect_preview)
	dock.add_child(disconnect_button)
	status = Label.new()
	status.text = "Disconnected"
	dock.add_child(status)
	add_control_to_dock(DOCK_SLOT_RIGHT_UL, dock)

func _exit_tree() -> void:
	disconnect_preview()
	remove_control_from_docks(dock)
	dock.queue_free()

func connect_preview(code: String) -> void:
	var parts := code.strip_edges().split(" ", false)
	var local := RegEx.create_from_string("^http://(?:127[.]0[.]0[.]1|\\[::1\\]):[0-9]+/api/editor-preview/adapter/godot/[A-Za-z0-9_-]+$")
	if parts.size() != 2 or local.search(parts[0]) == null:
		status.text = "Paste the Godot pairing code from Unsloth Studio."
		return
	disconnect_preview()
	endpoint = parts[0]
	secret = parts[1]
	instance_id = Crypto.new().generate_random_bytes(16).hex_encode()
	pairing.clear()
	status.text = "Connecting"
	EditorInterface.set_main_screen_editor("3D")

func disconnect_preview() -> void:
	endpoint = ""
	secret = ""
	active = false
	input_busy = false
	frame_busy = false
	last_camera = null
	input_request.cancel_request()
	frame_request.cancel_request()
	status.text = "Disconnected"

func _headers() -> PackedStringArray:
	return PackedStringArray(["Authorization: Bearer " + secret, "X-Unsloth-Preview-Instance: " + instance_id])

func _process(delta: float) -> void:
	if endpoint.is_empty():
		return
	elapsed += delta
	if elapsed < INTERVAL:
		return
	elapsed = 0.0
	if not input_busy:
		input_busy = input_request.request(endpoint + "/input", _headers()) == OK
	if active and not frame_busy:
		_capture()

func _capture() -> void:
	var viewport := EditorInterface.get_editor_viewport_3d()
	if not viewport:
		return
	var container := viewport.get_parent() as CanvasItem
	if container and not container.is_visible_in_tree():
		return
	var image := viewport.get_texture().get_image()
	if not image or image.is_empty():
		return
	var size := image.get_size()
	var scale := minf(1.0, minf(1280.0 / size.x, 720.0 / size.y))
	if scale < 1.0:
		image.resize(maxi(1, roundi(size.x * scale)), maxi(1, roundi(size.y * scale)), Image.INTERPOLATE_BILINEAR)
	var headers := _headers()
	headers.append("Content-Type: image/jpeg")
	frame_busy = frame_request.request_raw(endpoint + "/frame", headers, HTTPClient.METHOD_POST,
		image.save_jpg_to_buffer(0.76)) == OK

func _receive_frame(_result: int, code: int, _headers_value: PackedStringArray, _body: PackedByteArray) -> void:
	frame_busy = false
	if code == 401 or code == 409:
		disconnect_preview()
		status.text = "Start a new preview in Studio"

func _receive_input(result: int, code: int, _headers_value: PackedStringArray, body: PackedByteArray) -> void:
	input_busy = false
	if code == 401 or code == 409:
		disconnect_preview()
		status.text = "Start a new preview in Studio"
		return
	if result != HTTPRequest.RESULT_SUCCESS or code != 200:
		active = false
		status.text = "Waiting for Studio"
		return
	var data: Variant = JSON.parse_string(body.get_string_from_utf8())
	if not data is Dictionary:
		return
	active = bool(data.get("active"))
	status.text = "Live" if active else "Waiting for Studio"
	if active and (data.orbit_x != 0 or data.orbit_y != 0 or data.pan_x != 0 or data.pan_y != 0 or data.zoom != 0):
		_navigate(data)

func _navigate(delta: Dictionary) -> void:
	var viewport := EditorInterface.get_editor_viewport_3d()
	var camera := viewport.get_camera_3d() if viewport else null
	if not camera:
		return
	var transform := camera.global_transform
	if camera != last_camera or not transform.is_equal_approx(last_transform):
		last_camera = camera
		pivot = transform.origin - transform.basis.z * maxf(0.5, transform.origin.length())
	var original := transform.origin - pivot
	var distance := maxf(0.05, original.length())
	var offset := original.rotated(Vector3.UP, -float(delta.orbit_x) * PI)
	offset = offset.rotated(transform.basis.x, -float(delta.orbit_y) * PI)
	if absf(offset.normalized().dot(Vector3.UP)) >= 0.995:
		offset = original
	pivot += (transform.basis.y * float(delta.pan_y) - transform.basis.x * float(delta.pan_x)) * distance * 2
	offset = offset.normalized() * clampf(distance * exp(float(delta.zoom) * 1.7), 0.05, 10000)
	camera.look_at_from_position(pivot + offset, pivot)
	last_transform = camera.global_transform
