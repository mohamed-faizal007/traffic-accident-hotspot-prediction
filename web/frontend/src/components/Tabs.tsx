export default function Tabs<T extends string>({
  label,
  value,
  tabs,
  onChange,
}: {
  label: string;
  value: T;
  tabs: { id: T; text: string }[];
  onChange: (id: T) => void;
}) {
  return (
    <div role="tablist" aria-label={label} className="flex flex-wrap gap-2 border-b border-border">
      {tabs.map((t) => {
        const active = t.id === value;
        return (
          <button
            key={t.id}
            role="tab"
            id={`tab-${t.id}`}
            aria-selected={active}
            aria-controls={`panel-${t.id}`}
            type="button"
            onClick={() => onChange(t.id)}
            className={`-mb-px border-b-2 px-4 py-2.5 text-base font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
              active ? "border-accent text-text-primary" : "border-transparent text-text-secondary hover:text-text-primary"
            }`}
          >
            {t.text}
          </button>
        );
      })}
    </div>
  );
}
