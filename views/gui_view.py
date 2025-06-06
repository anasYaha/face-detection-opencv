import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
from controllers.enrollment_controller import EnrollmentController
from controllers.recognition_controller import RecognitionController
from repositories.user_repository import UserRepository
from services.file_service import FileService


class FaceRecognitionGUI:

    def __init__(self, enrollment_controller: EnrollmentController,
                 recognition_controller: RecognitionController,
                 user_repository: UserRepository):
        self.enrollment_controller = enrollment_controller
        self.recognition_controller = recognition_controller
        self.user_repository = user_repository

        self.root = tk.Tk()
        self.root.title("Face Recognition System")
        self.root.geometry("450x350")

        self._setup_gui()

    def _setup_gui(self) -> None:
        # Title
        title_label = tk.Label(self.root, text="Face Recognition System",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=20)

        # Enroll button
        enroll_btn = tk.Button(self.root, text="Enroll New Face",
                               command=self._on_enroll_click,
                               bg="#4CAF50", fg="white",
                               font=("Arial", 12), width=25, height=2)
        enroll_btn.pack(pady=10)

        # Camera button - FIX: Updated text to reflect new functionality
        camera_btn = tk.Button(self.root, text="Start Camera (Detection + Recognition)",
                               command=self._on_camera_click,
                               bg="#2196F3", fg="white",
                               font=("Arial", 12), width=25, height=2)
        camera_btn.pack(pady=10)

        # View enrolled faces button
        view_btn = tk.Button(self.root, text="View Enrolled Faces",
                             command=self._on_view_click,
                             bg="#FF9800", fg="white",
                             font=("Arial", 12), width=25, height=2)
        view_btn.pack(pady=10)

        # FIX: Add settings button for threshold adjustment
        settings_btn = tk.Button(self.root, text="Adjust Recognition Settings",
                                 command=self._on_settings_click,
                                 bg="#9C27B0", fg="white",
                                 font=("Arial", 10), width=25, height=1)
        settings_btn.pack(pady=5)

        # Status label
        self.status_label = tk.Label(self.root, text=self._get_status_text(),
                                     font=("Arial", 10))
        self.status_label.pack(pady=10)

        # FIX: Add info label
        info_label = tk.Label(self.root,
                              text="Camera works even without enrolled faces!\n"
                                   "Blue = Detection only, Green = Recognized, Red = Unknown",
                              font=("Arial", 8), fg="gray")
        info_label.pack(pady=5)

    def _get_status_text(self) -> str:
        user_count = len(self.user_repository.get_all_users())
        if user_count == 0:
            return "No faces enrolled - Camera will work in detection-only mode"
        return f"Enrolled faces: {user_count} - Camera will work with recognition"

    def _on_enroll_click(self) -> None:
        try:
            first_name = simpledialog.askstring("First Name", "Enter first name:")
            if not first_name:
                return

            last_name = simpledialog.askstring("Last Name", "Enter last name:")
            if not last_name:
                return

            age_str = simpledialog.askstring("Age", "Enter age:")
            if not age_str:
                return

            try:
                age = int(age_str)
            except ValueError:
                messagebox.showerror("Error", "Age must be a number")
                return

            # Show instructions
            messagebox.showinfo("Instructions",
                                "Position your face in the camera.\n"
                                "Press SPACE to capture samples.\n"
                                "We need 5 samples for good recognition.\n"
                                "Press ESC to cancel enrollment.")

            success = self.enrollment_controller.enroll_user(first_name, last_name, age)

            if success:
                messagebox.showinfo("Success", f"Face enrolled successfully for {first_name} {last_name}")
                self._update_status()
            else:
                messagebox.showerror("Error", "Failed to enroll face")

        except Exception as e:
            messagebox.showerror("Error", f"Enrollment failed: {str(e)}")

    def _on_camera_click(self) -> None:
        try:
            user_count = len(self.user_repository.get_all_users())
            if user_count == 0:
                messagebox.showinfo("Camera Starting",
                                    "Starting camera in detection-only mode.\n"
                                    "Faces will be detected but not recognized.\n"
                                    "Enroll faces to enable recognition!")
            else:
                messagebox.showinfo("Camera Starting",
                                    f"Starting camera with {user_count} enrolled faces.\n"
                                    "Recognition will be active!")

            self.recognition_controller.start_recognition()
        except Exception as e:
            messagebox.showerror("Error", f"Camera failed: {str(e)}")

    def _on_settings_click(self) -> None:
        try:
            current_threshold = self.recognition_controller.face_service.recognition_threshold

            threshold_str = simpledialog.askstring(
                "Recognition Settings",
                f"Current recognition threshold: {current_threshold}\n"
                f"Lower values = stricter recognition\n"
                f"Recommended range: 50-150\n"
                f"Enter new threshold:"
            )

            if threshold_str:
                try:
                    new_threshold = float(threshold_str)
                    if 10 <= new_threshold <= 200:
                        self.recognition_controller.face_service.set_recognition_threshold(new_threshold)
                        messagebox.showinfo("Success", f"Recognition threshold set to {new_threshold}")
                    else:
                        messagebox.showerror("Error", "Threshold must be between 10 and 200")
                except ValueError:
                    messagebox.showerror("Error", "Please enter a valid number")

        except Exception as e:
            messagebox.showerror("Error", f"Settings update failed: {str(e)}")

    def _on_view_click(self) -> None:
        users = self.user_repository.get_all_users()
        if not users:
            messagebox.showinfo("Info", "No faces enrolled yet.\nCamera still works in detection-only mode!")
            return

        self._show_enrolled_faces_window(users)

    def _show_enrolled_faces_window(self, users) -> None:
        view_window = tk.Toplevel(self.root)
        view_window.title("Enrolled Faces")
        view_window.geometry("700x500")

        # Create scrollable frame
        canvas = tk.Canvas(view_window)
        scrollbar = ttk.Scrollbar(view_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Add user information
        for user in users.values():
            self._create_user_frame(scrollable_frame, user, view_window)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _create_user_frame(self, parent, user, view_window) -> None:
        frame = ttk.Frame(parent, relief="ridge", borderwidth=2)
        frame.pack(fill="x", padx=10, pady=5)

        # Display face image
        if user.face_files:
            try:
                img = Image.open(user.face_files[0])
                img.thumbnail((80, 80))
                photo = ImageTk.PhotoImage(img)
                img_label = ttk.Label(frame, image=photo)
                img_label.image = photo  # Keep reference
                img_label.pack(side="left", padx=10, pady=5)
            except Exception as e:
                print(f"Error loading image: {e}")

        # User info
        info_frame = ttk.Frame(frame)
        info_frame.pack(side="left", fill="x", expand=True, padx=5)

        info_text = (f"ID: {user.id} | Name: {user.full_name} | "
                     f"Age: {user.age} | Enrolled: {user.enrolled_date}")
        label = ttk.Label(info_frame, text=info_text, padding=5)
        label.pack(anchor="w")

        samples_text = f"Samples: {len(user.face_files)}"
        samples_label = ttk.Label(info_frame, text=samples_text, padding=5)
        samples_label.pack(anchor="w")

        # Delete button
        delete_btn = tk.Button(
            frame,
            text="Delete",
            command=lambda: self._delete_user(user.id, view_window),
            bg="#FF5252",
            fg="white"
        )
        delete_btn.pack(side="right", padx=10, pady=5)

    def _delete_user(self, user_id: int, view_window) -> None:
        user = self.user_repository.get_user(user_id)
        if not user:
            messagebox.showerror("Error", "User not found!")
            return

        if messagebox.askyesno(
                "Confirm Deletion",
                f"Are you sure you want to delete {user.full_name}?\n"
                "This action cannot be undone!"
        ):
            try:
                # Delete user files and data
                file_service = FileService()
                file_service.delete_user_files(user_id)
                self.user_repository.delete_user(user_id)

                self._update_status()
                view_window.destroy()
                self._on_view_click()  # Refresh view

                messagebox.showinfo("Success", f"Deleted {user.full_name} successfully")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete user: {str(e)}")

    def _update_status(self) -> None:

        self.status_label.config(text=self._get_status_text())

    def run(self) -> None:
        self.root.mainloop()