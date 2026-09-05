import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { useAuth } from "../context/AuthContext"
import api from "../services/api"

function StatCard({ label, value, sub, color }) {
  return (
    <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5">
      <div className="text-xs text-slate-500 dark:text-slate-400 mb-2">{label}</div>
      <div className={`text-3xl font-medium ${color || ""}`}>{value}</div>
      {sub && <div className="text-xs text-slate-400 dark:text-slate-500 mt-1">{sub}</div>}
    </div>
  )
}

function Badge({ label, type }) {
  const cls = {
    IMAGE: "bg-blue-50 text-blue-700 dark:bg-blue-950/40 dark:text-blue-300",
    VIDEO: "bg-red-50 text-red-700 dark:bg-red-950/40 dark:text-red-300",
    AUDIO: "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300",
    FAKE: "bg-red-50 text-red-700 dark:bg-red-950/40 dark:text-red-300",
    REAL: "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300",
  }
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium ${cls[type] || cls[label]}`}>
      {label}
    </span>
  )
}

export default function Dashboard() {
  const { t } = useTranslation()
  const { user } = useAuth()
  const [history, setHistory] = useState([])
  const [admins, setAdmins] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      api.get("/analysis/history").catch(() => ({ data: [] })),
      user?.role === "SUPER_ADMIN"
        ? api.get("/admins/").catch(() => ({ data: [] }))
        : Promise.resolve({ data: [] }),
    ]).then(([h, a]) => {
      setHistory(Array.isArray(h.data) ? h.data : [])
      setAdmins(Array.isArray(a.data) ? a.data : [])
    }).finally(() => setLoading(false))
  }, [user])

  const total = history.length
  const fakes = history.filter((h) => h.label === "FAKE").length
  const reals = history.filter((h) => h.label === "REAL").length
  const recent = history.slice(0, 5)
  const images = history.filter((h) => h.media_type === "IMAGE").length
  const videos = history.filter((h) => h.media_type === "VIDEO").length
  const audios = history.filter((h) => h.media_type === "AUDIO").length

  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-medium">
          {t("greeting")}, {user?.full_name?.split(" ")[0] || "Admin"}
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">{t("dashSub")}</p>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-40">
          <div className="w-6 h-6 border-2 border-red-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
            <StatCard label={t("totalAnalyses")} value={total} />
            <StatCard label={t("detectedFakes")} value={fakes}
              sub={total ? `${((fakes / total) * 100).toFixed(1)}%` : "0%"}
              color="text-red-600 dark:text-red-400" />
            <StatCard label={t("realContent")} value={reals}
              sub={total ? `${((reals / total) * 100).toFixed(1)}%` : "0%"}
              color="text-emerald-600 dark:text-emerald-400" />
            <StatCard label={t("activeAdmins")} value={admins.filter((a) => a.is_active).length}
              color="text-green-700 dark:text-green-400" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5">
              <div className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-4">{t("recentAnalyses")}</div>
              {recent.length === 0 ? (
                <p className="text-sm text-slate-400 dark:text-slate-500">{t("noHistory")}</p>
              ) : (
                <div className="space-y-2">
                  {recent.map((item, i) => (
                    <div key={i} className="flex items-center gap-2 p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800">
                      <Badge label={item.media_type || "?"} type={item.media_type} />
                      <span className="text-sm flex-1 truncate">{item.original_name || item.media_file_id}</span>
                      <Badge label={item.label || "?"} type={item.label} />
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5">
              <div className="text-sm font-medium text-slate-500 dark:text-slate-400 mb-4">{t("distributionType")}</div>
              <div className="space-y-4">
                {[
                  { label: "Images", count: images, color: "bg-blue-500" },
                  { label: "Vidéos", count: videos, color: "bg-red-600" },
                  { label: "Audios", count: audios, color: "bg-green-700" },
                ].map(({ label, count, color }) => (
                  <div key={label}>
                    <div className="flex justify-between text-sm mb-1.5">
                      <span>{label}</span>
                      <span className="text-slate-400 dark:text-slate-500">
                        {count} ({total ? ((count / total) * 100).toFixed(1) : 0}%)
                      </span>
                    </div>
                    <div className="h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${color} rounded-full transition-all`}
                        style={{ width: total ? `${(count / total) * 100}%` : "0%" }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}