import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import {
  Bot, BookOpen, Phone, PhoneCall, MessageSquare, BarChart3, ShieldCheck, Bell,
  CreditCard, Settings, ChevronsUpDown, HelpCircle, Sparkles, Wrench,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { isSupabaseConfigured, supabase } from "@/lib/supabase";

const nav = [
  {
    label: "BUILD",
    items: [
      { to: "/agents", label: "Agents", icon: Bot },
      { to: "/knowledge-base", label: "Knowledge Base", icon: BookOpen },
      { to: "/tools", label: "Tools", icon: Wrench },
    ],
  },
  {
    label: "DEPLOY",
    items: [
      { to: "/phone-numbers", label: "Phone Numbers", icon: Phone },
      { to: "/batch-call", label: "Batch Call", icon: PhoneCall },
    ],
  },
  {
    label: "MONITOR",
    items: [
      { to: "/call-history", label: "Call History", icon: PhoneCall },
      { to: "/chat-history", label: "Chat History", icon: MessageSquare },
      { to: "/analytics", label: "Analytics", icon: BarChart3 },
      { to: "/quality", label: "AI Quality Assurance", icon: ShieldCheck },
      { to: "/alerting", label: "Alerting", icon: Bell },
    ],
  },
  {
    label: "SYSTEM",
    items: [
      { to: "/billing", label: "Billing", icon: CreditCard },
      { to: "/settings", label: "Settings", icon: Settings },
    ],
  },
];

const Sidebar = () => (
  <aside className="hidden md:flex w-64 shrink-0 flex-col border-r border-border bg-surface">
    <div className="px-5 pt-5 pb-3 flex items-center gap-2">
      <div className="w-8 h-8 rounded-lg bg-gradient-primary flex items-center justify-center shadow-elegant">
        <Sparkles className="w-4 h-4 text-primary-foreground" />
      </div>
      <span className="font-semibold tracking-tight text-lg">Vocal</span>
    </div>
    <button className="mx-3 mb-3 flex items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm hover:bg-secondary transition">
      <div className="w-5 h-5 rounded bg-primary/10 text-primary flex items-center justify-center text-[10px] font-bold">C</div>
      <span className="flex-1 text-left truncate">Christos … Workspace</span>
      <ChevronsUpDown className="w-3.5 h-3.5 text-muted-foreground" />
    </button>
    <nav className="flex-1 overflow-y-auto px-3 pb-4 space-y-5">
      {nav.map((group) => (
        <div key={group.label}>
          <div className="px-2 mb-1.5 text-[11px] font-semibold tracking-wider text-muted-foreground">
            {group.label}
          </div>
          <ul className="space-y-0.5">
            {group.items.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  className={({ isActive }) =>
                    cn(
                      "flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition",
                      isActive
                        ? "bg-accent text-accent-foreground font-medium"
                        : "text-foreground/75 hover:bg-secondary hover:text-foreground"
                    )
                  }
                >
                  <item.icon className="w-4 h-4" />
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
    <div className="border-t border-border p-3 space-y-2">
      <div className="flex items-center justify-between rounded-lg bg-gradient-soft px-3 py-2 text-sm">
        <span className="flex items-center gap-1.5 font-medium text-accent-foreground">
          <Sparkles className="w-3.5 h-3.5" /> Free Trial
        </span>
        <ChevronsUpDown className="w-3.5 h-3.5 text-muted-foreground" />
      </div>
      <div className="flex items-center gap-2 rounded-lg px-2 py-1.5">
        <div className="w-7 h-7 rounded-full bg-gradient-primary text-primary-foreground flex items-center justify-center text-xs font-semibold">
          C
        </div>
        <span className="flex-1 text-sm truncate">c.papaharalabou…</span>
        <ChevronsUpDown className="w-3.5 h-3.5 text-muted-foreground" />
      </div>
      <button className="w-full flex items-center gap-2 px-2 py-1.5 text-xs text-muted-foreground hover:text-foreground">
        <HelpCircle className="w-3.5 h-3.5" /> Help
      </button>
    </div>
  </aside>
);

export default function AppLayout() {
  const location = useLocation();
  const navigate = useNavigate();
  const [checkingAuth, setCheckingAuth] = useState(true);
  const isFlow = /^\/agents\/[^/]+/.test(location.pathname);

  useEffect(() => {
    let mounted = true;

    async function checkAuth() {
      if (!isSupabaseConfigured()) {
        if (mounted) setCheckingAuth(false);
        return;
      }
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (!mounted) return;
      if (!session) {
        navigate("/auth", { replace: true, state: { from: location.pathname } });
        return;
      }
      setCheckingAuth(false);
    }

    setCheckingAuth(true);
    void checkAuth();

    return () => {
      mounted = false;
    };
  }, [location.pathname, navigate]);

  if (checkingAuth) {
    return <div className="flex min-h-screen items-center justify-center bg-background text-sm text-muted-foreground">Checking session...</div>;
  }

  return (
    <div className="min-h-screen flex bg-background text-foreground">
      {!isFlow && <Sidebar />}
      <main className="flex-1 min-w-0 flex flex-col">
        <Outlet />
      </main>
    </div>
  );
}
