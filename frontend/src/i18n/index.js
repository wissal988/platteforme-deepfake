import i18n from "i18next"
import { initReactI18next } from "react-i18next"
import fr from "./fr.json"
import en from "./en.json"
import ar from "./ar.json"

const savedLang = localStorage.getItem("lang") || "fr"

i18n.use(initReactI18next).init({
  resources: {
    fr: { translation: fr },
    en: { translation: en },
    ar: { translation: ar },
  },
  lng: savedLang,
  fallbackLng: "fr",
  interpolation: { escapeValue: false },
})

if (savedLang === "ar") {
  document.documentElement.setAttribute("dir", "rtl")
} else {
  document.documentElement.setAttribute("dir", "ltr")
}

export default i18n