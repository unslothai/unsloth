-- SPDX-License-Identifier: AGPL-3.0-only
-- Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

local Http = game:GetService("HttpService")
local Capture = game:GetService("StudioCaptureService")
local RunService = game:GetService("RunService")
local toolbar = plugin:CreateToolbar("Unsloth")
local toggle = toolbar:CreateButton("Live Preview", "Connect to local Unsloth Studio", "")
local dock = plugin:CreateDockWidgetPluginGuiAsync("UnslothLivePreview",
    DockWidgetPluginGuiInfo.new(Enum.InitialDockState.Right, false, true, 320, 140, 250, 105))
dock.Title = "Unsloth Live Preview"

local layout = Instance.new("UIListLayout")
layout.Padding = UDim.new(0, 6)
layout.Parent = dock
local code = Instance.new("TextBox")
code.PlaceholderText = "Pairing code from Unsloth Studio"
code.Text = ""
code.ClearTextOnFocus = false
code.Size = UDim2.new(1, 0, 0, 32)
code.Parent = dock
local button = Instance.new("TextButton")
button.Text = "Connect preview"
button.Size = UDim2.new(1, 0, 0, 32)
button.Parent = dock
local status = Instance.new("TextLabel")
status.Text = "Disconnected"
status.Size = UDim2.new(1, 0, 0, 26)
status.Parent = dock
local generation = 0
local connected = false

local function disconnect(message)
    generation += 1
    connected = false
    button.Text = "Connect preview"
    status.Text = message or "Disconnected"
end

local function navigate(delta)
    local camera = workspace.CurrentCamera
    if not camera then return end
    if delta.orbit_x == 0 and delta.orbit_y == 0 and delta.pan_x == 0 and delta.pan_y == 0 and delta.zoom == 0 then
        return
    end
    local view = camera.CFrame
    local pivot = camera.Focus.Position
    local offset = view.Position - pivot
    if offset.Magnitude < 0.05 then
        offset = -view.LookVector * 10
        pivot = view.Position - offset
    end
    local distance = offset.Magnitude
    local rotation = CFrame.fromAxisAngle(Vector3.yAxis, -delta.orbit_x * math.pi)
        * CFrame.fromAxisAngle(view.RightVector, -delta.orbit_y * math.pi)
    local rotated = rotation:VectorToWorldSpace(offset)
    if math.abs(rotated.Unit:Dot(Vector3.yAxis)) < 0.995 then offset = rotated end
    pivot += (view.UpVector * delta.pan_y - view.RightVector * delta.pan_x) * distance * 2
    offset = offset.Unit * math.clamp(distance * math.exp(delta.zoom * 1.7), 0.05, 10000)
    camera.CFrame = CFrame.lookAt(pivot + offset, pivot)
    camera.Focus = CFrame.new(pivot)
end

local function frame(session)
    if not Capture:CanCaptureScreenshot() then return nil end
    local camera = workspace.CurrentCamera
    if not camera then return nil end
    local size = camera.ViewportSize
    if size.X < 1 or size.Y < 1 then return nil end
    local scale = math.min(1, 1280 / size.X, 720 / size.Y)
    local capture = Capture:CaptureScreenshot({
        OutputSize = Vector2.new(math.max(1, math.floor(size.X * scale)), math.max(1, math.floor(size.Y * scale))),
        ResampleMode = Enum.ResamplerMode.Default,
        Format = Enum.StudioCaptureScreenshotFormat.PNG,
        UICaptureMode = Enum.UICaptureMode.None,
    })
    if not capture then return nil end
    local deadline = os.clock() + 0.5
    while session == generation and os.clock() < deadline and not RunService:IsRunning() do
        local state = capture.BufferStatus
        if state == Enum.StudioCaptureBufferStatus.Ready then
            local bytes = capture:GetBuffer()
            if bytes and buffer.len(bytes) <= 4 * 1024 * 1024 then return buffer.tostring(bytes) end
            return nil
        elseif state == Enum.StudioCaptureBufferStatus.Error then
            return nil
        end
        task.wait()
    end
    return nil
end

local function connect()
    local endpoint, token = code.Text:match("^(%S+)%s+([%w_-]+)%s*$")
    if not endpoint or not (
        endpoint:match("^http://127%.0%.0%.1:%d+/api/editor%-preview/adapter/roblox/[%w_-]+$")
        or endpoint:match("^http://%[::1%]:%d+/api/editor%-preview/adapter/roblox/[%w_-]+$")
    ) then
        status.Text = "Paste the Roblox Studio pairing code."
        return
    end
    generation += 1
    local session = generation
    local instance = Http:GenerateGUID(false):gsub("-", ""):lower()
    connected = true
    code.Text = ""
    button.Text = "Disconnect"
    status.Text = "Connecting"
    task.spawn(function()
        local ok, granted = pcall(function()
            return Capture:CanCaptureScreenshot() or Capture:RequestScreenshotPermissionAsync()
        end)
        if session ~= generation then return end
        if not ok or not granted then
            disconnect("Allow viewport capture and reconnect.")
            return
        end
        local function request(image)
            local headers = { Authorization = "Bearer " .. token, ["X-Unsloth-Preview-Instance"] = instance }
            if image then headers["Content-Type"] = "image/png" end
            return Http:RequestAsync({
                Url = endpoint .. (if image then "/exchange" else "/input"),
                Method = if image then "POST" else "GET",
                Headers = headers,
                Body = image,
                Timeout = 1,
            })
        end
        local streaming = false
        while session == generation do
            local started = os.clock()
            local image
            if streaming and not RunService:IsRunning() then
                local ready, result = pcall(frame, session)
                if ready then image = result end
            end
            if session ~= generation then return end
            local sent, response = pcall(request, image)
            if session ~= generation then return end
            local delay = 0.5
            if sent and (response.StatusCode == 401 or response.StatusCode == 409) then
                disconnect("Start a new preview in Studio.")
                return
            elseif sent and response.StatusCode == 429 then
                streaming = false
                status.Text = "Roblox is limiting preview requests"
                delay = 2
            elseif sent and response.Success then
                local decoded, delta = pcall(function() return Http:JSONDecode(response.Body) end)
                if decoded then
                    streaming = delta.active and not RunService:IsRunning()
                    if streaming then
                        navigate(delta)
                        status.Text = if image then "Live" else "Waiting for a viewport frame"
                        delay = if image then 1 / 15 else 0.25
                    else
                        status.Text = if RunService:IsRunning() then "Return to edit mode" else "Waiting for Studio"
                    end
                end
            else
                streaming = false
                status.Text = "Waiting for Studio"
            end
            task.wait(math.max(0.01, delay - (os.clock() - started)))
        end
    end)
end

button.Activated:Connect(function()
    if connected then disconnect() else connect() end
end)
toggle.Click:Connect(function() dock.Enabled = not dock.Enabled end)
plugin.Unloading:Connect(function() disconnect() end)
