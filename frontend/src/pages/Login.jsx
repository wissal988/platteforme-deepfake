import { useState } from "react"
import { useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { Eye, EyeOff } from "lucide-react"
import { useAuth } from "../context/AuthContext"
import LanguageSwitcher from "../components/layout/LanguageSwitcher"
import ThemeToggle from "../components/layout/ThemeToggle"
import entvLogo from "../assets/entv-logo.png"

export default function Login() {
  const { t } = useTranslation()
  const { login } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [showPwd, setShowPwd] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError("")
    setLoading(true)
    try {
      await login(email, password)
      navigate("/dashboard")
    } catch {
      setError(t("loginError"))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-100 dark:bg-slate-950 flex items-center justify-center p-4">
      <div className="absolute top-4 right-4 flex gap-2">
        <LanguageSwitcher />
        <ThemeToggle />
      </div>

      <div className="w-full max-w-sm">
        <div className="flex flex-col items-center mb-8">
          <div className="mb-4 p-2 bg-white dark:bg-slate-900 rounded-2xl shadow border border-slate-200 dark:border-slate-700">
            <img src={entvLogo} alt="ENTV" className="h-14 w-auto object-contain" />
          </div>
          <h1 className="text-xl font-semibold text-red-700 dark:text-red-400">{t("loginTitle")}</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1 text-center">{t("loginSub")}</p>
        </div>

        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6">
          {error && (
            <div className="mb-4 px-4 py-3 rounded-xl bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-900 text-red-700 dark:text-red-300 text-sm">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1.5 text-slate-700 dark:text-slate-200">{t("email")}</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="w-full px-3 py-2.5 rounded-xl border border-slate-200 dark:border-slate-600 bg-slate-50 dark:bg-slate-800 text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-transparent transition"
                placeholder="admin@exemple.com"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1.5 text-slate-700 dark:text-slate-200">{t("password")}</label>
              <div className="relative">
                <input
                  type={showPwd ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="w-full px-3 py-2.5 pr-10 rounded-xl border border-slate-200 dark:border-slate-600 bg-slate-50 dark:bg-slate-800 text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-transparent transition"
                  placeholder="••••••••"
                />
                <button
                  type="button"
                  onClick={() => setShowPwd(!showPwd)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
                >
                  {showPwd ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 rounded-xl bg-red-600 hover:bg-red-700 text-white text-sm font-medium transition disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {loading ? t("loggingIn") : t("loginBtn")}
            </button>
          </form>
        </div>

        <p className="text-center text-xs text-green-700 dark:text-green-500 mt-4 font-medium">{t("appSub")}</p>
      </div>
    </div>
  )
}