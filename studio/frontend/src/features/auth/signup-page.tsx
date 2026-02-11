import { LightRays } from "@/components/ui/light-rays";
import { AuthForm } from "./components/auth-form";

export function SignupPage() {
  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-background px-6 py-10 md:px-10">
      <LightRays
        count={6}
        color="rgba(34, 197, 94, 0.25)"
        blur={34}
        speed={15}
        length="70vh"
        style={{ opacity: 0.4 }}
      />
      <div className="relative z-10 w-full max-w-sm">
        <AuthForm mode="signup" />
      </div>
    </div>
  );
}
