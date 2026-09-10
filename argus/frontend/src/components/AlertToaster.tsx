import { useAlertCenter, TIER_HEX } from "../lib/alerts";

export default function AlertToaster() {
  const { toasts, dismissToast, muted, setMuted } = useAlertCenter();

  return (
    <div className="fixed bottom-4 right-4 z-50 w-80 space-y-2">
      {/* Global mute — visible whenever the alert system is mounted */}
      <div className="flex justify-end">
        <button
          onClick={() => setMuted(!muted)}
          title={muted ? "Unmute alerts" : "Mute alerts"}
          className="border border-argus-border bg-argus-card px-2 py-1 text-xs text-gray-400 hover:bg-argus-border hover:text-white"
        >
          {muted ? "🔇 Alerts muted" : "🔔 Alerts on"}
        </button>
      </div>

      {toasts.map((t) => {
        const color = TIER_HEX[t.event.tier] || TIER_HEX.Low;
        const time = t.event.timestamp ? new Date(t.event.timestamp).toLocaleTimeString() : "";
        return (
          <div
            key={t.id}
            role="alert"
            aria-live="assertive"
            className="border border-argus-border bg-argus-card p-3 shadow-lg"
            style={{ borderLeft: `4px solid ${color}` }}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold" style={{ color }}>
                {t.event.tier}
              </span>
              <button
                onClick={() => dismissToast(t.id)}
                className="text-xs text-gray-500 hover:text-white"
                aria-label="Dismiss alert"
              >
                ✕
              </button>
            </div>
            <div className="mt-1 text-sm text-gray-100">
              {t.event.behaviour_type.replace(/_/g, " ")}
              {time && <span className="ml-2 text-xs text-gray-400">{time}</span>}
            </div>
            {t.text && (
              <p className="mt-1 text-xs text-gray-400">
                {t.text}
                {t.translated && <span className="ml-1 italic">({t.language})</span>}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
