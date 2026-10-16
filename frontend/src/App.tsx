import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { ToastProvider } from "./components/ui";
import Confidence from "./pages/Confidence";
import Dashboard from "./pages/Dashboard";
import Experiments from "./pages/Experiments";
import HistoryPage from "./pages/History";
import Landing from "./pages/Landing";
import Results from "./pages/Results";
import SettingsPage from "./pages/SettingsPage";
import Training from "./pages/Training";
import UploadPage from "./pages/Upload";

export default function App() {
  return (
    <ToastProvider>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/app" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="upload" element={<UploadPage />} />
          <Route path="results/:id" element={<Results />} />
          <Route path="confidence" element={<Confidence />} />
          <Route path="training" element={<Training />} />
          <Route path="experiments" element={<Experiments />} />
          <Route path="history" element={<HistoryPage />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </ToastProvider>
  );
}
