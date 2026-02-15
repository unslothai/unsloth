export function ReadMore({ href = "#" }: { href?: string }) {
  return (
    <a
      href={href}
      className="text-emerald-600 underline underline-offset-2 hover:text-emerald-700"
    >
      Read more
    </a>
  );
}

