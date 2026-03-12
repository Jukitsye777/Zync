import { useToast } from "@/hooks/use-toast";
import { Toast, ToastClose, ToastDescription, ToastProvider, ToastTitle, ToastViewport } from "@/components/ui/toast";

export function Toaster() {
  const { toasts } = useToast();

  return (
    <ToastProvider>
      {toasts.map(function ({ id, title, description, action, ...props }) {
        return (
          <Toast key={id} {...props} 
          className={`pointer-events-auto fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 
w-[480px] max-w-[95vw] rounded-3xl shadow-2xl p-8 transition-all text-lg ${
  props.variant === "destructive"
    ? "bg-red-50 border border-red-300 text-red-900"
    : "bg-blue-50 border border-blue-300 text-blue-900"
}`}

>
            <div className="grid gap-1 pr-6">

              {title && (
  <ToastTitle className="text-xl font-bold tracking-tight">

    {title}
  </ToastTitle>
)}
              {description && (
  <ToastDescription className="text-base opacity-90">

    {description}
  </ToastDescription>
)}

            </div>
            {action}
            <ToastClose className="absolute top-3 right-3 opacity-70 hover:opacity-100" />

          </Toast>
        );
      })}
      <ToastViewport className="fixed inset-0 z-[100] pointer-events-none" />

    </ToastProvider>
  );
}
