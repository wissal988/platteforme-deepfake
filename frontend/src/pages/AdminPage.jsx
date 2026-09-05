import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { Plus, X, Pencil } from "lucide-react"
import api from "../services/api"

export default function AdminPage() {
  const { t } = useTranslation()
  const [admins, setAdmins] = useState([])
  const [loading, setLoading] = useState(true)
  const [showModal, setShowModal] = useState(false)
  const [editAdmin, setEditAdmin] = useState(null)
  const [form, setForm] = useState({ full_name: "", email: "", password: "" })
  const [formLoading, setFormLoading] = useState(false)
  const [formError, setFormError] = useState("")

  const load = () => {
    api.get("/admins/")
      .then((res) => setAdmins(Array.isArray(res.data) ? res.data : []))
      .catch(() => setAdmins([]))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  const openCreate = () => {
    setEditAdmin(null)
    setForm({ full_name: "", email: "", password: "" })
    setFormError("")
    setShowModal(true)
  }

  const openEdit = (admin) => {
    setEditAdmin(admin)
    setForm({ full_name: admin.full_name, email: admin.email, password: "" })
    setFormError("")
    setShowModal(true)
  }

  const submitForm = async (e) => {
    e.preventDefault()
    setFormError("")
    setFormLoading(true)
    try {
      if (editAdmin) {
        const payload = {}
        if (form.full_name !== editAdmin.full_name) payload.full_name = form.full_name
        if (form.email !== editAdmin.email) payload.email = form.email
        if (form.password) payload.password = form.password
        await api.patch(`/admins/${editAdmin.id}`, payload)
      } else {
        await api.post("/admins/", form)
      }
      setShowModal(false)
      load()
    } catch (err) {
      setFormError(err?.response?.data?.detail || t("errorLoad"))
    } finally {
      setFormLoading(false)
    }
  }

  const toggleActive = async (id, current) => {
    try {
      await api.patch(`/admins/${id}/${current ? "deactivate" : "activate"}`)
      load()
    } catch (_) {}
  }

  const deleteAdmin = async (id) => {
    if (!window.confirm("Supprimer cet admin ?")) return
    try {
      await api.delete(`/admins/${id}`)
      load()
    } catch (_) {}
  }

  const initials = (name) =>
    name?.split(" ").map((w) => w[0]).join("").toUpperCase().slice(0, 2) || "AD"

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-xl font-medium">{t("adminTitle")}</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">{t("adminSub")}</p>
        </div>
        <button
          onClick={openCreate}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-red-600 hover:bg-red-700 text-white text-sm font-medium transition"
        >
          <Plus size={15} />
          {t("addAdmin")}
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="w-6 h-6 border-2 border-red-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : (
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {admins.map((admin) => (
            <div
              key={admin.id}
              className={`bg-white dark:bg-slate-900 rounded-2xl border p-5 ${
                admin.is_active
                  ? "border-slate-200 dark:border-slate-800"
                  : "border-slate-100 dark:border-slate-800 opacity-60"
              }`}
            >
              <div className="flex items-center gap-3 mb-4">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium flex-shrink-0 ${
                  admin.is_active
                    ? "bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300"
                    : "bg-slate-100 dark:bg-slate-800 text-slate-500"
                }`}>
                  {initials(admin.full_name)}
                </div>
                <div className="min-w-0">
                  <div className="text-sm font-medium truncate">{admin.full_name}</div>
                  <div className="text-xs text-slate-400 dark:text-slate-500 truncate">{admin.email}</div>
                </div>
              </div>

              <div className="flex items-center gap-1.5 mb-4">
                <div className={`w-1.5 h-1.5 rounded-full ${admin.is_active ? "bg-emerald-500" : "bg-red-400"}`} />
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  {admin.is_active ? t("active") : t("inactive")}
                </span>
              </div>

              <div className="flex gap-2 pt-3 border-t border-slate-100 dark:border-slate-800 flex-wrap">
                <button
                  onClick={() => openEdit(admin)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-xs hover:bg-slate-50 dark:hover:bg-slate-800 transition"
                >
                  <Pencil size={11} />
                  {t("editAdmin")}
                </button>
                <button
                  onClick={() => toggleActive(admin.id, admin.is_active)}
                  className="flex-1 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-xs hover:bg-slate-50 dark:hover:bg-slate-800 transition"
                >
                  {admin.is_active ? t("deactivate") : t("activate")}
                </button>
                <button
                  onClick={() => deleteAdmin(admin.id)}
                  className="flex-1 py-1.5 rounded-lg border border-slate-100 dark:border-slate-800 text-xs text-red-500 hover:bg-red-50 dark:hover:bg-red-950/20 hover:border-red-200 dark:hover:border-red-900 transition"
                >
                  {t("deleteAdmin")}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40">
          <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 w-full max-w-md p-6 shadow-xl">
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-base font-medium">
                {editAdmin ? t("editAdmin") : t("addAdmin")}
              </h2>
              <button
                onClick={() => { setShowModal(false); setFormError("") }}
                className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
              >
                <X size={18} />
              </button>
            </div>

            {formError && (
              <div className="mb-4 px-4 py-3 rounded-xl bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-900 text-red-700 dark:text-red-300 text-sm">
                {formError}
              </div>
            )}

            <form onSubmit={submitForm} className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1.5">{t("fullName")}</label>
                <input
                  type="text"
                  value={form.full_name}
                  onChange={(e) => setForm({ ...form, full_name: e.target.value })}
                  required
                  className="w-full px-3 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-sm focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-transparent transition"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1.5">{t("email")}</label>
                <input
                  type="email"
                  value={form.email}
                  onChange={(e) => setForm({ ...form, email: e.target.value })}
                  required
                  className="w-full px-3 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-sm focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-transparent transition"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1.5">
                  {t("password")}
                  {editAdmin && <span className="text-xs text-slate-400 ml-2">(laisser vide pour ne pas changer)</span>}
                </label>
                <input
                  type="password"
                  value={form.password}
                  onChange={(e) => setForm({ ...form, password: e.target.value })}
                  required={!editAdmin}
                  className="w-full px-3 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-sm focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-transparent transition"
                  placeholder={editAdmin ? "••••••••" : t("passwordPlaceholder")}
                />
              </div>
              <div className="flex gap-3 pt-1">
                <button
                  type="button"
                  onClick={() => { setShowModal(false); setFormError("") }}
                  className="flex-1 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 text-sm hover:bg-slate-50 dark:hover:bg-slate-800 transition"
                >
                  {t("cancel")}
                </button>
                <button
                  type="submit"
                  disabled={formLoading}
                  className="flex-1 py-2.5 rounded-xl bg-red-600 hover:bg-red-700 text-white text-sm font-medium transition disabled:opacity-50"
                >
                  {formLoading ? "..." : t("save")}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}