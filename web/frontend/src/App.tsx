import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import ActionList from "./pages/ActionList";
import NavBar from "./components/NavBar";
import Patterns from "./pages/Patterns";
import RegressionTrack from "./pages/RegressionTrack";
import FeatureImportance from "./pages/FeatureImportance";
import Home from "./pages/Home";
import ModelPerformance from "./pages/ModelPerformance";
import RiskMap from "./pages/RiskMap";

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-bg">
        <NavBar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/action-list" element={<ActionList />} />
          <Route path="/predictions" element={<Navigate to="/action-list" replace />} />
          <Route path="/risk-map" element={<RiskMap />} />
          <Route path="/patterns" element={<Patterns />} />
          <Route path="/regression-track" element={<RegressionTrack />} />
          <Route path="/model-performance" element={<ModelPerformance />} />
          <Route path="/feature-importance" element={<FeatureImportance />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
