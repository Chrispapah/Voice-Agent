import { LucideIcon } from "lucide-react";

export default function PageStub({
  title, description, icon: Icon,
}: { title: string; description: string; icon: LucideIcon }) {
  return (
    <div className="flex flex-col flex-1">
      <header className="px-8 py-5 border-b border-border">
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
      </header>
      <div className="flex-1 grid place-items-center px-8 py-16">
        <div className="max-w-md text-center">
          <div className="w-14 h-14 mx-auto rounded-2xl bg-gradient-primary shadow-elegant flex items-center justify-center text-primary-foreground">
            <Icon className="w-6 h-6" />
          </div>
          <h2 className="mt-5 text-lg font-semibold">{title}</h2>
          <p className="mt-2 text-sm text-muted-foreground">{description}</p>
        </div>
      </div>
    </div>
  );
}
