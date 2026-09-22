export interface Column<T> {
  key: string;
  header: string;
  align?: "left" | "right";
  render: (row: T) => React.ReactNode;
}

export default function DataTable<T>({ columns, rows, rowKey }: { columns: Column<T>[]; rows: T[]; rowKey: (row: T) => string }) {
  return (
    <div className="overflow-hidden rounded-card border border-border">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-bg-raised">
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key}
                  className={`px-4 py-3 font-medium text-text-secondary ${col.align === "right" ? "text-right" : "text-left"}`}
                >
                  {col.header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={rowKey(row)} className="border-t border-gridline bg-card hover:bg-card-hover">
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={`px-4 py-2.5 tabular-nums text-text-primary ${col.align === "right" ? "text-right" : "text-left"}`}
                  >
                    {col.render(row)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length === 0 && <div className="p-8 text-center text-sm text-text-muted">No rows match the current filters.</div>}
    </div>
  );
}
