import flet as ft
import tensorflow as tf
from PIL import Image

def main(page: ft.Page):
    page.title = "Pneumonia Checker"
    page.window_width = 700
    page.window_height = 350
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    def pick_file_and_predict(e: ft.FilePickerResultEvent):
        model = tf.keras.models.load_model(r"assets\model_files\model.h5")
        prediction.value = ""
        result_file = e.files
        path_to_image = result_file[0].path

        image.src = result_file[0].path
        image.update()

        selected_files.value = (f"file path: {path_to_image}")
        selected_files.update()

        image_tf = tf.keras.preprocessing.image.load_img(path_to_image)
        image_tf = tf.image.resize(image_tf, size=(512, 512))
        image_tf = tf.reshape(image_tf, (512, 512, 3))
        image_tf = tf.expand_dims(image_tf, axis=0)
        image_prediction = model.predict(image_tf)[0]

        if image_prediction[1]>image_prediction[0] and image_prediction[1]<0.7:
            prediction.value = f"RESULT: probably pneumonia ({round(float(image_prediction[1]), 1) * 100}% pneumonia chance)"
        elif image_prediction[1]>image_prediction[0] and image_prediction[1]>0.7:
            prediction.value = f"RESULT: pneumonia ({round(float(image_prediction[1]), 1) * 100}% pneumonia chance)"
        else:
            prediction.value = f"RESULT: clear ({round(float(image_prediction[1]), 1) * 100}% pneumonia chance)"
        prediction.update()

    
    pick_files_dialog = ft.FilePicker(on_result=pick_file_and_predict)
    selected_files = ft.Text("no_image_imported")
    image = ft.Image(src=r"assets\no_image.png",width=100,height=100,fit=ft.ImageFit.CONTAIN)
    prediction = ft.Text()
    
    page.overlay.append(pick_files_dialog)

    page.add(
        ft.Row([
                ft.ElevatedButton("import image", icon=ft.icons.UPLOAD_FILE,
                on_click=lambda _: pick_files_dialog.pick_files()),
                ], alignment=ft.MainAxisAlignment.CENTER),
        ft.Divider(),
        ft.Row([image, selected_files], alignment=ft.MainAxisAlignment.CENTER),
        ft.Divider(),
        ft.Row([prediction], alignment=ft.MainAxisAlignment.CENTER),
    )
    

ft.app(target=main)