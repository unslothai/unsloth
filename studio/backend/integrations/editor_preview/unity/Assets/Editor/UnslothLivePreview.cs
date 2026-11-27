// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

using System;
using System.Text.RegularExpressions;
using UnityEditor;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Networking;
using UnityEngine.Rendering;

internal sealed class UnslothLivePreview : EditorWindow
{
    [Serializable]
    private sealed class Gesture
    {
        public bool active;
        public float orbit_x, orbit_y, pan_x, pan_y, zoom;
    }

    private string pairing = "", endpoint = "", token = "", status = "Disconnected";
    private string instance;
    private SceneView viewport;
    private UnityWebRequest input, upload;
    private double nextPoll, nextFrame;
    private int generation;
    private bool streaming, reading;

    [MenuItem("Window/Unsloth/Live Preview")]
    private static void Open() => GetWindow<UnslothLivePreview>("Unsloth Preview");

    private void OnEnable()
    {
        EditorApplication.update += Tick;
        Camera.onPostRender += Capture;
        RenderPipelineManager.endCameraRendering += CaptureSrp;
    }

    private void OnDisable()
    {
        EditorApplication.update -= Tick;
        Camera.onPostRender -= Capture;
        RenderPipelineManager.endCameraRendering -= CaptureSrp;
        Disconnect();
    }

    private void OnGUI()
    {
        EditorGUILayout.LabelField("Unsloth Studio Live Preview", EditorStyles.boldLabel);
        EditorGUILayout.LabelField(status);
        if (endpoint.Length == 0)
        {
            pairing = EditorGUILayout.PasswordField("Pairing code", pairing);
            if (GUILayout.Button("Connect preview")) Connect();
        }
        else if (GUILayout.Button("Disconnect")) Disconnect();
    }

    private void Connect()
    {
        var parts = pairing.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length != 2 || !Regex.IsMatch(parts[0],
            @"\Ahttp://(?:127\.0\.0\.1|\[::1\]):[0-9]+/api/editor-preview/adapter/unity/[A-Za-z0-9_-]+\z"))
        {
            status = "Paste the Unity pairing code from Unsloth Studio.";
            return;
        }
        viewport = SceneView.lastActiveSceneView;
        if (viewport == null || !SystemInfo.supportsAsyncGPUReadback)
        {
            status = "Open a Scene view on a graphics backend that supports GPU readback.";
            return;
        }
        endpoint = parts[0];
        token = parts[1];
        instance = Guid.NewGuid().ToString("N");
        pairing = "";
        status = "Connecting";
    }

    private void Disconnect()
    {
        generation++;
        streaming = false;
        endpoint = token = "";
        status = "Disconnected";
        input?.Dispose();
        upload?.Dispose();
        input = upload = null;
    }

    private void Send(UnityWebRequest request)
    {
        request.redirectLimit = 0;
        request.timeout = 1;
        request.SetRequestHeader("Authorization", "Bearer " + token);
        request.SetRequestHeader("X-Unsloth-Preview-Instance", instance);
        request.SendWebRequest();
    }

    private void Tick()
    {
        if (endpoint.Length == 0) return;
        if (input != null && input.isDone)
        {
            var request = input;
            input = null;
            using (request)
            {
                if (request.responseCode == 401 || request.responseCode == 409)
                {
                    Disconnect();
                    status = "Start a new preview in Studio";
                    Repaint();
                    return;
                }
                var gesture = request.result == UnityWebRequest.Result.Success
                    ? JsonUtility.FromJson<Gesture>(request.downloadHandler.text) : null;
                streaming = gesture != null && gesture.active;
                var nextStatus = streaming ? "Live" : "Waiting for Studio";
                if (viewport == null) nextStatus = "The selected Scene view was closed";
                if (status != nextStatus) { status = nextStatus; Repaint(); }
                if (streaming && viewport != null) Navigate(gesture);
            }
        }
        if (upload != null && upload.isDone)
        {
            upload.Dispose();
            upload = null;
        }
        var now = EditorApplication.timeSinceStartup;
        if (input == null && now >= nextPoll)
        {
            nextPoll = now + 1.0 / 20;
            input = UnityWebRequest.Get(endpoint + "/input");
            Send(input);
        }
        if (streaming && viewport != null && now >= nextFrame && upload == null && !reading)
            viewport.Repaint();
    }

    private void Navigate(Gesture gesture)
    {
        if (gesture.orbit_x == 0 && gesture.orbit_y == 0 && gesture.pan_x == 0 && gesture.pan_y == 0 && gesture.zoom == 0)
            return;
        var rotation = viewport.rotation;
        var pivot = viewport.pivot + rotation * new Vector3(-gesture.pan_x, gesture.pan_y, 0) * viewport.size * 2;
        var yaw = Quaternion.AngleAxis(gesture.orbit_x * 180, Vector3.up);
        var pitch = Quaternion.AngleAxis(gesture.orbit_y * 180, Vector3.right);
        var size = Mathf.Clamp(viewport.size * Mathf.Exp(gesture.zoom * 1.7f), .01f, 10000f);
        viewport.LookAtDirect(pivot, yaw * rotation * pitch, size);
    }

    private void CaptureSrp(ScriptableRenderContext _, Camera camera) => Capture(camera);

    private void Capture(Camera camera)
    {
        var now = EditorApplication.timeSinceStartup;
        if (!streaming || viewport == null || viewport.camera != camera || reading || upload != null || now < nextFrame)
            return;
        var source = camera.targetTexture;
        if (source == null || source.width < 1 || source.height < 1) return;
        nextFrame = now + 1.0 / 15;
        var scale = Mathf.Min(1f, 1280f / source.width, 720f / source.height);
        var width = Mathf.Max(1, Mathf.RoundToInt(source.width * scale));
        var height = Mathf.Max(1, Mathf.RoundToInt(source.height * scale));
        var target = RenderTexture.GetTemporary(width, height, 0, RenderTextureFormat.ARGB32);
        var previous = RenderTexture.active;
        Graphics.Blit(source, target);
        RenderTexture.active = previous;
        var session = generation;
        reading = true;
        AsyncGPUReadback.Request(target, 0, TextureFormat.RGBA32, result =>
        {
            try
            {
                if (result.hasError || generation != session || !streaming || upload != null) return;
                using (var jpeg = ImageConversion.EncodeNativeArrayToJPG(result.GetData<byte>(),
                    GraphicsFormat.R8G8B8A8_UNorm, (uint)width, (uint)height, 0, 76))
                {
                    upload = new UnityWebRequest(endpoint + "/frame", "POST");
                    upload.uploadHandler = new UploadHandlerRaw(jpeg.ToArray()) { contentType = "image/jpeg" };
                    upload.downloadHandler = new DownloadHandlerBuffer();
                    Send(upload);
                }
            }
            finally
            {
                reading = false;
                RenderTexture.ReleaseTemporary(target);
            }
        });
    }
}
