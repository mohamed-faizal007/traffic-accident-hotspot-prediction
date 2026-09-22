import { Route, BrowserRouter, Routes } from "react-router-dom";
import NavBar from "./components/NavBar";
import FeatureImportance from "./pages/FeatureImportance";
import Home from "./pages/Home";
import ModelPerformance from "./pages/ModelPerformance";
import Predictions from "./pages/Predictions";
import RiskMap from "./pages/RiskMap";

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-bg">
        <NavBar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/predictions" element={<Predictions />} />
          <Route path="/risk-map" element={<RiskMap />} />
          <Route path="/model-performance" element={<ModelPerformance />} />
          <Route path="/feature-importance" element={<FeatureImportance />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
