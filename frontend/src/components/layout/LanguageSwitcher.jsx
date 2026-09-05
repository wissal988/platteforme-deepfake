import i18n from "i18next"

export default function LanguageSwitcher() {
  const change = (lang) => {
    i18n.changeLanguage(lang)
    localStorage.setItem("lang", lang)
    document.documentElement.setAttribute("dir", lang === "ar" ? "rtl" : "ltr")
    document.documentElement.setAttribute("lang", lang)
  }

  return (
    <select
      defaultValue={i18n.language}
      onChange={(e) => change(e.target.value)}
      className="text-xs px-2 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 cursor-pointer"
    >
      <option value="fr">FR</option>
      <option value="en">EN</option>
      <option value="ar">AR</option>
    </select>
  )
}