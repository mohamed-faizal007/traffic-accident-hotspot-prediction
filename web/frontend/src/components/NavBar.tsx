import { NavLink } from "react-router-dom";

const LINKS = [
  { to: "/", label: "Home" },
  { to: "/predictions", label: "Predictions" },
  { to: "/risk-map", label: "Risk Map" },
  { to: "/model-performance", label: "Model Performance" },
  { to: "/feature-importance", label: "Feature Importance" },
];

export default function NavBar() {
  return (
    <header className="sticky top-0 z-50 border-b border-border bg-bg-raised/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6 lg:px-16">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-tier-critical" />
          <span className="font-display text-[15px] font-semibold text-text-primary">Hotspot Prediction</span>
        </div>
        <nav className="flex items-center gap-6">
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              className={({ isActive }) =>
                `relative py-1 text-sm font-medium transition-colors ${
                  isActive ? "text-text-primary" : "text-text-secondary hover:text-text-primary"
                }`
              }
            >
              {({ isActive }) => (
                <>
                  {link.label}
                  {isActive && (
                    <span className="absolute -bottom-[1px] left-0 h-[2px] w-full rounded-pill bg-accent" />
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
