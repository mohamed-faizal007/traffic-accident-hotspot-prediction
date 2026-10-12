import { useState } from "react";
import { NavLink } from "react-router-dom";

const LINKS = [
  { to: "/", label: "Home" },
  { to: "/risk-map", label: "Risk Map" },
  { to: "/action-list", label: "Action list" },
  { to: "/patterns", label: "Patterns" },
  { to: "/regression-track", label: "Regression track" },
  { to: "/model-performance", label: "Model Performance" },
  { to: "/feature-importance", label: "Feature Importance" },
];

export default function NavBar() {
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-bg-raised/90 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-x-6 px-4 sm:px-6 md:h-16 md:flex-nowrap lg:px-16">
        <div className="flex h-14 items-center gap-2 md:h-auto">
          <span className="h-2 w-2 rounded-full bg-tier-critical" />
          <span className="font-display text-[15px] font-semibold text-text-primary">Hotspot Prediction</span>
        </div>

        <button
          type="button"
          aria-expanded={open}
          aria-controls="main-nav"
          onClick={() => setOpen((o) => !o)}
          className="rounded-sm border border-border bg-card px-3 py-1.5 text-sm font-medium text-text-primary hover:bg-card-hover focus:outline-none focus-visible:ring-2 focus-visible:ring-accent md:hidden"
        >
          {open ? "Close menu" : "Menu"}
        </button>

        <nav
          id="main-nav"
          aria-label="Main"
          className={`${open ? "flex" : "hidden"} w-full flex-col gap-1 pb-3 md:flex md:w-auto md:flex-row md:items-center md:gap-5 md:pb-0 lg:gap-6`}
        >
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === "/"}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                `relative whitespace-nowrap rounded-sm px-2 py-2.5 text-base font-medium transition-colors md:px-0 md:py-1 md:text-sm ${
                  isActive ? "text-text-primary max-md:bg-card" : "text-text-secondary hover:text-text-primary"
                }`
              }
            >
              {({ isActive }) => (
                <>
                  {link.label}
                  {isActive && (
                    <span className="absolute -bottom-[1px] left-0 hidden h-[2px] w-full rounded-pill bg-accent md:block" />
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
