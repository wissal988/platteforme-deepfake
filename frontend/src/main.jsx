import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import "./index.css"
import "./i18n"
import App from "./App.jsx"
import { ThemeProvider } from "./context/ThemeContext"
import { AuthProvider } from "./context/AuthContext"

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <ThemeProvider>
      <AuthProvider>
        <App />
      </AuthProvider>
    </ThemeProvider>
  </StrictMode>
)