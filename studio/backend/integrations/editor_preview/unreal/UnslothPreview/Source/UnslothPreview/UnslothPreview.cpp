// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

#include "Async/Async.h"
#include "Containers/Ticker.h"
#include "Editor.h"
#include "EditorViewportClient.h"
#include "HAL/IConsoleManager.h"
#include "HAL/PlatformApplicationMisc.h"
#include "HttpModule.h"
#include "IAssetViewport.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "ImageUtils.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "Internationalization/Regex.h"
#include "Json.h"
#include "LevelEditor.h"
#include "Math/RotationMatrix.h"
#include "Misc/Guid.h"
#include "Modules/ModuleManager.h"
#include "UnrealClient.h"

class FUnslothPreviewModule final : public IModuleInterface
{
    FString Endpoint;
    FString Token;
    FString Instance;
    FHttpRequestPtr InputRequest;
    FHttpRequestPtr UploadRequest;
    TFuture<TArray<uint8>> Encoding;
    TWeakPtr<IAssetViewport> SelectedViewport;
    FTSTicker::FDelegateHandle Ticker;
    IConsoleObject* ConnectCommand = nullptr;
    IConsoleObject* DisconnectCommand = nullptr;
    bool bStreaming = false;
    float Distance = 500.f;

    FHttpRequestRef Request(const TCHAR* Path, const TCHAR* Verb) const
    {
        auto Result = FHttpModule::Get().CreateRequest();
        Result->SetURL(Endpoint + Path);
        Result->SetVerb(Verb);
        Result->SetHeader(TEXT("Authorization"), TEXT("Bearer ") + Token);
        Result->SetHeader(TEXT("X-Unsloth-Preview-Instance"), Instance);
        Result->SetTimeout(1.f);
        return Result;
    }

    void Connect()
    {
        FString Pairing, Address, Secret;
        FPlatformApplicationMisc::ClipboardPaste(Pairing);
        Pairing.TrimStartAndEndInline();
        if (!Pairing.Split(TEXT(" "), &Address, &Secret) || Secret.IsEmpty()) return;
        const FRegexPattern Local(TEXT("^http://(127[.]0[.]0[.]1|\\[::1\\]):[0-9]+/api/editor-preview/adapter/unreal/[A-Za-z0-9_-]+$"));
        FRegexMatcher Match(Local, Address);
        if (!Match.FindNext() || Match.GetMatchEnding() != Address.Len())
        {
            UE_LOG(LogTemp, Warning, TEXT("Copy the Unreal pairing code from Unsloth Studio first."));
            return;
        }
        auto& LevelEditor = FModuleManager::LoadModuleChecked<FLevelEditorModule>("LevelEditor");
        TSharedPtr<IAssetViewport> Viewport = LevelEditor.GetFirstActiveViewport();
        if (!Viewport || !Viewport->GetAssetViewportClient().IsPerspective())
        {
            UE_LOG(LogTemp, Warning, TEXT("Open a perspective Level Editor viewport."));
            return;
        }
        Disconnect();
        SelectedViewport = Viewport;
        Endpoint = MoveTemp(Address);
        Token = MoveTemp(Secret);
        Instance = FGuid::NewGuid().ToString(EGuidFormats::Digits).ToLower();
        Ticker = FTSTicker::GetCoreTicker().AddTicker(
            FTickerDelegate::CreateRaw(this, &FUnslothPreviewModule::Tick), 1.f / 15.f);
    }

    void Disconnect()
    {
        if (Ticker.IsValid())
        {
            FTSTicker::GetCoreTicker().RemoveTicker(Ticker);
            Ticker.Reset();
        }
        for (FHttpRequestPtr* Pending : { &InputRequest, &UploadRequest })
        {
            if (!*Pending) continue;
            (*Pending)->OnProcessRequestComplete().Unbind();
            (*Pending)->CancelRequest();
            Pending->Reset();
        }
        if (Encoding.IsValid()) Encoding.Get();
        SelectedViewport.Reset();
        Endpoint.Reset();
        Token.Reset();
        bStreaming = false;
        Distance = 500.f;
    }

    bool Tick(float)
    {
        const auto Viewport = SelectedViewport.Pin();
        if (!Viewport || !GEditor || GEditor->PlayWorld) return true;
        if (!InputRequest)
        {
            InputRequest = Request(TEXT("/input"), TEXT("GET"));
            InputRequest->OnProcessRequestComplete().BindRaw(this, &FUnslothPreviewModule::ReceiveInput);
            if (!InputRequest->ProcessRequest()) InputRequest.Reset();
        }
        if (Encoding.IsValid() && Encoding.IsReady())
        {
            auto Body = Encoding.Get();
            Encoding = TFuture<TArray<uint8>>();
            if (bStreaming && !UploadRequest && Body.Num() > 0 && Body.Num() <= 4 * 1024 * 1024)
            {
                UploadRequest = Request(TEXT("/frame"), TEXT("POST"));
                UploadRequest->SetHeader(TEXT("Content-Type"), TEXT("image/jpeg"));
                UploadRequest->SetContent(MoveTemp(Body));
                UploadRequest->OnProcessRequestComplete().BindLambda(
                    [this](FHttpRequestPtr, FHttpResponsePtr, bool) { UploadRequest.Reset(); });
                if (!UploadRequest->ProcessRequest()) UploadRequest.Reset();
            }
        }
        if (bStreaming && !Encoding.IsValid() && !UploadRequest)
        {
            if (FViewport* Surface = Viewport->GetActiveViewport()) Capture(*Surface);
        }
        return true;
    }

    void ReceiveInput(FHttpRequestPtr, FHttpResponsePtr Response, bool bSuccess)
    {
        InputRequest.Reset();
        if (Response && (Response->GetResponseCode() == 401 || Response->GetResponseCode() == 409))
        {
            Disconnect();
            UE_LOG(LogTemp, Display, TEXT("Start a new preview in Unsloth Studio."));
            return;
        }
        TSharedPtr<FJsonObject> Data;
        bStreaming = bSuccess && Response && Response->GetResponseCode() == 200
            && FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Response->GetContentAsString()), Data)
            && Data.IsValid() && Data->GetBoolField(TEXT("active"));
        const auto Viewport = SelectedViewport.Pin();
        if (!bStreaming || !Viewport || !GEditor || GEditor->PlayWorld) return;

        const double X = Data->GetNumberField(TEXT("orbit_x"));
        const double Y = Data->GetNumberField(TEXT("orbit_y"));
        const double PanX = Data->GetNumberField(TEXT("pan_x"));
        const double PanY = Data->GetNumberField(TEXT("pan_y"));
        const double Zoom = Data->GetNumberField(TEXT("zoom"));
        if (!X && !Y && !PanX && !PanY && !Zoom) return;
        FEditorViewportClient& Client = Viewport->GetAssetViewportClient();
        if (!Client.IsPerspective()) return;
        const FRotator Previous = Client.GetViewRotation();
        const FRotationMatrix Basis(Previous);
        FVector Pivot = Client.GetViewLocation() + Previous.Vector() * Distance;
        Pivot += (Basis.GetScaledAxis(EAxis::Z) * PanY - Basis.GetScaledAxis(EAxis::Y) * PanX) * Distance * 2;
        FRotator Rotation = Previous;
        Rotation.Yaw -= X * 180;
        Rotation.Pitch = FMath::Clamp(FRotator::NormalizeAxis(Rotation.Pitch - Y * 180), -89.0, 89.0);
        Distance = FMath::Clamp(Distance * FMath::Exp(static_cast<float>(Zoom) * 1.7f), 1.f, 100000.f);
        Client.SetViewRotation(Rotation);
        Client.SetViewLocation(Pivot - Rotation.Vector() * Distance);
        Client.Invalidate();
    }

    void Capture(FViewport& Viewport)
    {
        const FIntPoint Source = Viewport.GetSizeXY();
        if (Source.X < 1 || Source.Y < 1 || Source.X > 4096 || Source.Y > 4096) return;
        TArray<FColor> Pixels;
        if (!Viewport.ReadPixels(Pixels) || Pixels.Num() != Source.X * Source.Y) return;
        auto Wrapper = FModuleManager::LoadModuleChecked<IImageWrapperModule>("ImageWrapper").CreateImageWrapper(EImageFormat::JPEG);
        if (!Wrapper) return;
        Encoding = Async(EAsyncExecution::ThreadPool, [Source, Wrapper, Pixels = MoveTemp(Pixels)]() mutable {
            TArray<uint8> Result;
            const float Scale = FMath::Min3(1.f, 1280.f / Source.X, 720.f / Source.Y);
            const FIntPoint Size(FMath::Max(1, FMath::RoundToInt(Source.X * Scale)),
                                 FMath::Max(1, FMath::RoundToInt(Source.Y * Scale)));
            if (Size != Source)
            {
                TArray<FColor> Resized;
                FImageUtils::ImageResize(Source.X, Source.Y, Pixels, Size.X, Size.Y, Resized, true, true);
                Pixels = MoveTemp(Resized);
            }
            if (Wrapper->SetRaw(Pixels.GetData(), Pixels.Num() * sizeof(FColor),
                                Size.X, Size.Y, ERGBFormat::BGRA, 8))
            {
                const auto& Jpeg = Wrapper->GetCompressed(76);
                if (Jpeg.Num() <= 4 * 1024 * 1024) Result.Append(Jpeg.GetData(), static_cast<int32>(Jpeg.Num()));
            }
            return Result;
        });
    }

    void StartupModule() override
    {
        auto& Console = IConsoleManager::Get();
        ConnectCommand = Console.RegisterConsoleCommand(
            TEXT("UnslothPreview.Connect"), TEXT("Read the pairing code from the clipboard and connect."),
            FConsoleCommandDelegate::CreateRaw(this, &FUnslothPreviewModule::Connect));
        DisconnectCommand = Console.RegisterConsoleCommand(
            TEXT("UnslothPreview.Disconnect"), TEXT("Disconnect the live viewport."),
            FConsoleCommandDelegate::CreateRaw(this, &FUnslothPreviewModule::Disconnect));
    }

    void ShutdownModule() override
    {
        Disconnect();
        IConsoleManager::Get().UnregisterConsoleObject(ConnectCommand);
        IConsoleManager::Get().UnregisterConsoleObject(DisconnectCommand);
    }
};

IMPLEMENT_MODULE(FUnslothPreviewModule, UnslothPreview)
