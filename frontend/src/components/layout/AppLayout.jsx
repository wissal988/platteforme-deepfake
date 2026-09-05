import { Link, Outlet, useLocation } from "react-router-dom"
import { useTranslation } from "react-i18next"
import {
  LayoutDashboard, Upload, History, Users, LogOut
} from "lucide-react"
import { useAuth } from "../../context/AuthContext"
import ThemeToggle from "./ThemeToggle"
import LanguageSwitcher from "./LanguageSwitcher"
import entvLogo from "../../assets/entv-logo.png"

export default function AppLayout() {
  const { t } = useTranslation()
  const location = useLocation()
  const { user, logout } = useAuth()

  const nav = [
    { path: "/dashboard", label: t("dashboard"), icon: LayoutDashboard },
    { path: "/upload", label: t("upload"), icon: Upload },
    { path: "/history", label: t("history"), icon: History },
    ...(user?.role === "SUPER_ADMIN"
      ? [{ path: "/admin", label: t("admin"), icon: Users }]
      : []),
  ]

  const initials = user?.full_name
    ? user.full_name.split(" ").map((w) => w[0]).join("").toUpperCase().slice(0, 2)
    : "AD"

  return (
    <div className="min-h-screen flex bg-slate-100 dark:bg-slate-950 text-slate-900 dark:text-white">
      <aside className="w-64 flex-shrink-0 flex flex-col bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800">
        {/* Logo ENTV */}
        <div className="px-5 py-4 border-b border-slate-200 dark:border-slate-800">
          <div className="flex items-center gap-3">
            <img src={entvLogo} alt="ENTV" className="h-10 w-auto object-contain" />
            <div>
              <div className="text-sm font-semibold text-red-700 dark:text-red-400">{t("appName")}</div>
              <div className="text-xs text-green-700 dark:text-green-400">{t("appSub")}</div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 p-3 space-y-0.5">
          {nav.map(({ path, label, icon: Icon }) => {
            const active = location.pathname === path
            return (
              <Link
                key={path}
                to={path}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition ${
                  active
                    ? "bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 border-l-4 border-red-600"
                    : "text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-white"
                }`}
              >
                <Icon className={`w-4 h-4 flex-shrink-0 ${active ? "text-red-600" : ""}`} />
                {label}
              </Link>
            )
          })}
        </nav>

        {/* Controls + User */}
        <div className="p-3 border-t border-slate-200 dark:border-slate-800 space-y-2">
          <div className="flex gap-2">
            <LanguageSwitcher />
            <ThemeToggle />
          </div>
          <div className="flex items-center gap-2 p-2 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer">
            <div className="w-8 h-8 rounded-full bg-red-100 dark:bg-red-950 flex items-center justify-center text-xs font-medium text-red-700 dark:text-red-300 flex-shrink-0">
              {initials}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium truncate">{user?.full_name || "Admin"}</div>
              <div className="text-xs text-slate-500 dark:text-slate-400 truncate">{user?.role}</div>
            </div>
            <button
              onClick={logout}
              title={t("logout")}
              className="text-slate-400 hover:text-red-500 transition"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>

      <main className="flex-1 overflow-auto p-6 min-w-0">
        <Outlet />
      </main>
    </div>
  )
}