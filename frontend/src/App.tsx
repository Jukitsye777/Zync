import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './App.css'
import Index from './pages/index'
import NotFound from './pages/NotFound'
import { Toaster } from "@/components/ui/toaster";


function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Index />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
      <Toaster />
    </BrowserRouter>
  )
}

export default App
