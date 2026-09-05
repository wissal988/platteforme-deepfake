import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom"
import { useAuth } from "./context/AuthContext"
import AppLayout from "./components/layout/AppLayout"
import Login from "./pages/Login"
import Dashboard from "./pages/Dashboard"
import UploadPage from "./pages/UploadPage"
import HistoryPage from "./pages/HistoryPage"
import AdminPage from "./pages/AdminPage"

function ProtectedRoute({ children, requireSuperAdmin = false }) {
  const { user, loading } = useAuth()
  if (loading) return (
    <div className="min-h-screen flex items-center justify-center bg-slate-100 dark:bg-slate-950">
      <div className="w-8 h-8 border-2 border-violet-500 border-t-transparent rounded-full animate-spin" />
    </div>
  )
  if (!user) return <Navigate to="/login" replace />
  if (requireSuperAdmin && user.role !== "SUPER_ADMIN") return <Navigate to="/dashboard" replace />
  return children
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/login" replace />} />
        <Route path="/login" element={<Login />} />
        <Route element={
          <ProtectedRoute>
            <AppLayout />
          </ProtectedRoute>
        }>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/admin" element={
            <ProtectedRoute requireSuperAdmin>
              <AdminPage />
            </ProtectedRoute>
          } />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}