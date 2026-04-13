"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getMe } from "./api";
import { supabase } from "./supabase";

export interface User {
  id: string;
  email: string;
  display_name: string;
}

export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    getMe()
      .then(setUser)
      .catch(() => {
        router.push("/login");
      })
      .finally(() => setLoading(false));

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!session?.user) {
        setUser(null);
        router.push("/login");
        return;
      }
      setUser({
        id: session.user.id,
        email: session.user.email || "",
        display_name: session.user.user_metadata?.display_name || "",
      });
    });

    return () => subscription.unsubscribe();
  }, [router]);

  const logout = async () => {
    await supabase.auth.signOut();
    router.push("/login");
  };

  return { user, loading, logout };
}
