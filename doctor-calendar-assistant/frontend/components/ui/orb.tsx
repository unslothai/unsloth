"use client";

/**
 * Orb Component - ElevenLabs UI Style
 * 3D animated sphere using Three.js and React Three Fiber
 * Based on https://github.com/elevenlabs/ui
 */

import { useRef, useMemo, useEffect, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { cn } from "@/lib/utils";

// Agent states matching ElevenLabs API
type AgentState = "idle" | "listening" | "thinking" | "speaking";

interface OrbProps {
  state?: AgentState;
  colors?: [string, string];
  volume?: number;
  size?: number;
  className?: string;
}

// Vertex shader for the orb effect
const vertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

// Fragment shader with animated gradient and noise
const fragmentShader = `
  uniform float uTime;
  uniform float uVolume;
  uniform vec3 uColor1;
  uniform vec3 uColor2;
  uniform float uState; // 0: idle, 1: listening, 2: thinking, 3: speaking
  varying vec2 vUv;

  // Simplex noise function
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

  float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                        -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v -   i + dot(i, C.xx);
    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                            + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy),
                            dot(x12.zw, x12.zw)), 0.0);
    m = m*m;
    m = m*m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
  }

  void main() {
    vec2 center = vUv - 0.5;
    float dist = length(center);

    // Create circular gradient
    float circle = 1.0 - smoothstep(0.0, 0.5, dist);

    // Add noise for organic feel
    float noise = snoise(vUv * 3.0 + uTime * 0.5) * 0.5 + 0.5;

    // Volume-based pulsing
    float pulse = 1.0 + uVolume * 0.3;

    // State-based animation speed
    float speed = uState == 3.0 ? 2.0 : (uState == 2.0 ? 1.5 : 1.0);
    float breathe = sin(uTime * speed) * 0.1 + 1.0;

    // Mix colors based on noise and position
    vec3 color = mix(uColor1, uColor2, noise * circle);

    // Add glow
    float glow = pow(circle, 2.0) * pulse * breathe;
    color += glow * 0.3;

    // Edge fade
    float alpha = circle * circle;

    // Add rings for listening/speaking states
    if (uState >= 1.0) {
      float ring = sin(dist * 20.0 - uTime * 3.0) * 0.5 + 0.5;
      ring *= smoothstep(0.5, 0.3, dist) * smoothstep(0.0, 0.2, dist);
      color += ring * uVolume * 0.5;
    }

    gl_FragColor = vec4(color, alpha);
  }
`;

// Inner 3D sphere scene
function OrbScene({
  state,
  colors,
  volume,
}: {
  state: AgentState;
  colors: [string, string];
  volume: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);

  // Convert state to number for shader
  const stateValue = useMemo(() => {
    switch (state) {
      case "listening": return 1;
      case "thinking": return 2;
      case "speaking": return 3;
      default: return 0;
    }
  }, [state]);

  // Convert hex colors to THREE.Color
  const color1 = useMemo(() => new THREE.Color(colors[0]), [colors[0]]);
  const color2 = useMemo(() => new THREE.Color(colors[1]), [colors[1]]);

  // Shader uniforms
  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uVolume: { value: 0 },
      uColor1: { value: color1 },
      uColor2: { value: color2 },
      uState: { value: stateValue },
    }),
    []
  );

  // Update uniforms each frame
  useFrame((_, delta) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value += delta;

      // Smooth volume transition
      const targetVolume = volume;
      const currentVolume = materialRef.current.uniforms.uVolume.value;
      materialRef.current.uniforms.uVolume.value +=
        (targetVolume - currentVolume) * 0.1;

      // Update state
      materialRef.current.uniforms.uState.value = stateValue;

      // Update colors
      materialRef.current.uniforms.uColor1.value = color1;
      materialRef.current.uniforms.uColor2.value = color2;
    }

    // Rotate mesh slightly based on state
    if (meshRef.current) {
      const rotationSpeed = state === "thinking" ? 0.5 : 0.1;
      meshRef.current.rotation.z += delta * rotationSpeed;
    }
  });

  return (
    <mesh ref={meshRef}>
      <circleGeometry args={[1, 64]} />
      <shaderMaterial
        ref={materialRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
        transparent
      />
    </mesh>
  );
}

// Pulse rings around the orb
function PulseRings({ active, color }: { active: boolean; color: string }) {
  const ringsRef = useRef<THREE.Group>(null);
  const [rings] = useState(() => [0, 1, 2].map(() => ({ scale: 1, opacity: 0 })));

  useFrame((_, delta) => {
    if (!ringsRef.current || !active) return;

    rings.forEach((ring, i) => {
      ring.scale += delta * (1 + i * 0.3);
      ring.opacity = Math.max(0, 1 - (ring.scale - 1) / 0.5);

      if (ring.scale > 1.5) {
        ring.scale = 1;
        ring.opacity = 1;
      }
    });
  });

  if (!active) return null;

  return (
    <group ref={ringsRef}>
      {rings.map((ring, i) => (
        <mesh key={i} scale={ring.scale}>
          <ringGeometry args={[0.95, 1, 64]} />
          <meshBasicMaterial
            color={color}
            transparent
            opacity={ring.opacity * 0.3}
          />
        </mesh>
      ))}
    </group>
  );
}

// Main Orb component
export function Orb({
  state = "idle",
  colors = ["#3b82f6", "#8b5cf6"],
  volume = 0,
  size = 200,
  className,
}: OrbProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    // SSR placeholder
    return (
      <div
        className={cn(
          "rounded-full bg-gradient-to-br from-primary to-primary/50",
          className
        )}
        style={{ width: size, height: size }}
      />
    );
  }

  return (
    <div
      className={cn("relative", className)}
      style={{ width: size, height: size }}
    >
      <Canvas
        camera={{ position: [0, 0, 2], fov: 50 }}
        gl={{ alpha: true, antialias: true }}
        style={{ background: "transparent" }}
      >
        <OrbScene state={state} colors={colors} volume={volume} />
        <PulseRings
          active={state === "listening" || state === "speaking"}
          color={colors[0]}
        />
      </Canvas>
    </div>
  );
}

// Export state type for consumers
export type { AgentState };
