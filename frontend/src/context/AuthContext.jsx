import { createContext, useContext, useEffect, useState } from "react"
import api from "../services/api"

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const token = localStorage.getItem("token")
    if (!token) { setLoading(false); return }
    api.get("/auth/me")
      .then((res) => setUser(res.data))
      .catch(() => { localStorage.removeItem("token"); setUser(null) })
      .finally(() => setLoading(false))
  }, [])

  const login = async (email, password) => {
    const res = await api.post("/auth/login", { email, password })
    localStorage.setItem("token", res.data.access_token)
    setUser({ email: res.data.email, full_name: res.data.full_name, role: res.data.role })
    return res.data
  }

  const logout = () => {
    localStorage.removeItem("token")
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}