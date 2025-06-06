from repositories.user_repository import UserRepository
from services.file_service import FileService
from controllers.enrollment_controller import EnrollmentController
from controllers.recognition_controller import RecognitionController
from views.gui_view import FaceRecognitionGUI


def main():
    try:
        # Initialize repositories and services
        user_repository = UserRepository()
        file_service = FileService()

        # Initialize controllers
        enrollment_controller = EnrollmentController(user_repository, file_service)
        recognition_controller = RecognitionController(user_repository, file_service)

        # Initialize and run GUI
        app = FaceRecognitionGUI(enrollment_controller, recognition_controller, user_repository)
        app.run()

    except Exception as e:
        print(f"Application failed to start: {e}")


if __name__ == "__main__":
    main()