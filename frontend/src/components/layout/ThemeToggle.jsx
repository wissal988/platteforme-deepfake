import { Moon, Sun } from "lucide-react"
import { useTheme } from "../../context/ThemeContext"

export default function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()
  return (
    <button
      onClick={toggleTheme}
      className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-sm text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
    >
      {theme === "dark"
        ? <><Sun className="w-3.5 h-3.5" /> <span className="text-xs">Clair</span></>
        : <><Moon className="w-3.5 h-3.5" /> <span className="text-xs">Sombre</span></>
      }
    </button>
  )
}