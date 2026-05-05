import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "./pages/NotFound.tsx";
import AuthPage from "./pages/Auth.tsx";
import AgentsPage from "./pages/Agents.tsx";
import FlowBuilderPage from "./pages/FlowBuilder.tsx";
import KnowledgeBasePage from "./pages/KnowledgeBase.tsx";
import ToolsPage from "./pages/Tools.tsx";
import PhoneNumbersPage from "./pages/PhoneNumbers.tsx";
import CallHistoryPage from "./pages/CallHistory.tsx";
import AnalyticsPage from "./pages/Analytics.tsx";
import BillingPage from "./pages/Billing.tsx";
import BatchCallPage from "./pages/BatchCall.tsx";
import CreateBatchCallPage from "./pages/CreateBatchCall.tsx";
import ChatHistoryPage from "./pages/ChatHistory.tsx";
import QualityPage from "./pages/Quality.tsx";
import AlertingPage from "./pages/Alerting.tsx";
import SettingsPage from "./pages/Settings.tsx";
import AppLayout from "./components/layout/AppLayout.tsx";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/auth" element={<AuthPage />} />
          <Route element={<AppLayout />}>
            <Route path="/" element={<AgentsPage />} />
            <Route path="/agents" element={<AgentsPage />} />
            <Route path="/agents/:id" element={<FlowBuilderPage />} />
            <Route path="/knowledge-base" element={<KnowledgeBasePage />} />
            <Route path="/tools" element={<ToolsPage />} />
            <Route path="/phone-numbers" element={<PhoneNumbersPage />} />
            <Route path="/batch-call" element={<BatchCallPage />} />
            <Route path="/batch-call/new" element={<CreateBatchCallPage />} />
            <Route path="/call-history" element={<CallHistoryPage />} />
            <Route path="/chat-history" element={<ChatHistoryPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/quality" element={<QualityPage />} />
            <Route path="/alerting" element={<AlertingPage />} />
            <Route path="/billing" element={<BillingPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Route>
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
