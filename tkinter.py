from typing import Any, Tuple
import customtkinter as ctk

from icecream import ic


class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title('Window App')
        self.geometry(f"{1300}x{700}")
        self.minsize(1300, 700)

        self.sidebar_frame = Sidebar(master=self, corner_radius=0, width=200).pack(expand=False, fill='both', side='left')

        self.main_frame = MainFrame(master=self, corner_radius=0).pack(expand=True, fill='both', side='left')


class Sidebar(ctk.CTkFrame):
    def __init__(self, master: Any, width: int = 200, height: int = 200, corner_radius: int | str | None = None, border_width: int | str | None = None, bg_color: str | Tuple[str, str] = "transparent", fg_color: str | Tuple[str, str] | None = None, border_color: str | Tuple[str, str] | None = None, background_corner_colors: Tuple[str | Tuple[str, str]] | None = None, overwrite_preferred_drawing_method: str | None = None, **kwargs):
        super().__init__(master, width, height, corner_radius, border_width, bg_color, fg_color, border_color, background_corner_colors, overwrite_preferred_drawing_method, **kwargs)

        self.source_card = SourceCard( master=self, corner_radius=8 ).grid(row=0, column=0, padx=8, pady=(8,4), sticky='new') #.pack( expand=True, fill='x', side='top', padx=8, pady=(8,4) )

        self.info_card = InfoCard( master=self, corner_radius=8 ).grid(row=1, column=0, padx=8, pady=4, sticky='new') #.pack( expand=True, fill='x', side='top', padx=8, pady=(4,4) )

        self.model_card = ModelCard( master=self, corner_radius=8 ).grid(row=2, column=0, padx=8, pady=4, sticky='new') #.pack( expand=True, fill='x', side='top', padx=8, pady=(4,4) )

        self.options_card = OptionsCard( master=self, corner_radius=8 ).grid(row=3, column=0, padx=8, pady=8, sticky='sew') #.pack( expand=True, fill='x', side='top', padx=8, pady=(4,8) )


class SourceCard(ctk.CTkFrame):
    def __init__(self, master: Any, width: int = 200, height: int = 200, corner_radius: int | str | None = None, border_width: int | str | None = None, bg_color: str | Tuple[str] = "transparent", fg_color: str | Tuple[str] | None = None, border_color: str | Tuple[str] | None = None, background_corner_colors: Tuple[str | Tuple[str]] | None = None, overwrite_preferred_drawing_method: str | None = None, **kwargs):
        super().__init__(master, width, height, corner_radius, border_width, bg_color, fg_color, border_color, background_corner_colors, overwrite_preferred_drawing_method, **kwargs)

        self.source_label = ctk.CTkLabel(master=self, text='Source', font=ctk.CTkFont(size=20)).grid(row=0, column=0, padx=8, pady=8, sticky='w')

        self.source_add_button = ctk.CTkButton(master=self, text='Add', width=40).grid(row=1, column=0, padx=8, pady=8, sticky='e')


class InfoCard(ctk.CTkFrame):
    def __init__(self, master: Any, width: int = 200, height: int = 200, corner_radius: int | str | None = None, border_width: int | str | None = None, bg_color: str | Tuple[str] = "transparent", fg_color: str | Tuple[str] | None = None, border_color: str | Tuple[str] | None = None, background_corner_colors: Tuple[str | Tuple[str]] | None = None, overwrite_preferred_drawing_method: str | None = None, **kwargs):
        super().__init__(master, width, height, corner_radius, border_width, bg_color, fg_color, border_color, background_corner_colors, overwrite_preferred_drawing_method, **kwargs)

        self.info_label = ctk.CTkLabel(master=self, text='Information', font=ctk.CTkFont(size=20)).grid(row=0, column=0, padx=8, pady=(8,4), sticky="w")

        self.filename_value = ctk.CTkLabel(master=self, text = 'File Name').grid(row=1, column=0, padx=8, pady=(4,4), sticky="w")
        
        self.size_value = ctk.CTkLabel(master=self, text = 'Width X Height').grid(row=2, column=0, padx=8, pady=(4,4), sticky="w")
        
        self.total_value = ctk.CTkLabel(master=self, text = 'Total Frames').grid(row=3, column=0, padx=8, pady=(4,4), sticky="w")
        
        self.fps_value = ctk.CTkLabel(master=self, text = 'FPS').grid(row=4, column=0, padx=8, pady=(4,8), sticky="w")


class ModelCard(ctk.CTkFrame):
    def __init__(self, master: Any, width: int = 200, height: int = 200, corner_radius: int | str | None = None, border_width: int | str | None = None, bg_color: str | Tuple[str] = "transparent", fg_color: str | Tuple[str] | None = None, border_color: str | Tuple[str] | None = None, background_corner_colors: Tuple[str | Tuple[str]] | None = None, overwrite_preferred_drawing_method: str | None = None, **kwargs):
        super().__init__(master, width, height, corner_radius, border_width, bg_color, fg_color, border_color, background_corner_colors, overwrite_preferred_drawing_method, **kwargs)

        self.model_label = ctk.CTkLabel(master=self, text='Model', font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=8, pady=(4,4), sticky="w")

        self.model_start_button = ctk.CTkButton(master=self, width=70, text='Start').grid(row=1, column=0, padx=8, pady=(4,4), sticky="ew")

        self.model_stop_button = ctk.CTkButton(master=self, width=70, text='Stop').grid(row=1, column=1, padx=8, pady=(4,8), sticky="ew")


class OptionsCard(ctk.CTkFrame):
    def __init__(self, master: Any, width: int = 200, height: int = 200, corner_radius: int | str | None = None, border_width: int | str | None = None, bg_color: str | Tuple[str] = "transparent", fg_color: str | Tuple[str] | None = None, border_color: str | Tuple[str] | None = None, background_corner_colors: Tuple[str | Tuple[str]] | None = None, overwrite_preferred_drawing_method: str | None = None, **kwargs):
        super().__init__(master, width, height, corner_radius, border_width, bg_color, fg_color, border_color, background_corner_colors, overwrite_preferred_drawing_method, **kwargs)

        self.language_combobox = ctk.CTkComboBox(master=self, width=80, values=['ESP', 'ENG']).grid(row=0, column=0, padx=(8, 4), pady=(8, 8))

        self.theme_button = ctk.CTkButton(master=self, width=32, text='T').grid(row=0, column=1, padx=(4, 4), pady=(8, 8))
        
        self.about_button = ctk.CTkButton(master=self, width=32, text='A').grid(row=0, column=2, padx=(4, 8), pady=(8, 8))


class MainFrame(ctk.CTkFrame):
    def __init__(self, master: Any, width: int = 200, height: int = 200, corner_radius: int | str | None = None, border_width: int | str | None = None, bg_color: str | Tuple[str] = "transparent", fg_color: str | Tuple[str] | None = None, border_color: str | Tuple[str] | None = None, background_corner_colors: Tuple[str | Tuple[str]] | None = None, overwrite_preferred_drawing_method: str | None = None, **kwargs):
        super().__init__(master, width, height, corner_radius, border_width, bg_color, fg_color, border_color, background_corner_colors, overwrite_preferred_drawing_method, **kwargs)
            
        self.video_toolbar_card = VideoToolbarCard( master=self, corner_radius=8 ).pack( expand=False, fill='x', side='top', padx=8, pady=(8,4) )

        self.video_card = VideoCard(master=self,corner_radius=8 ).pack( expand=True, fill='both', side='top', padx=8, pady=(4,8) )
    
class VideoToolbarCard(ctk.CTkFrame):
    def __init__(self, master: Any, width: int = 200, height: int = 200, corner_radius: int | str | None = None, border_width: int | str | None = None, bg_color: str | Tuple[str] = "transparent", fg_color: str | Tuple[str] | None = None, border_color: str | Tuple[str] | None = None, background_corner_colors: Tuple[str | Tuple[str]] | None = None, overwrite_preferred_drawing_method: str | None = None, **kwargs):
        super().__init__(master, width, height, corner_radius, border_width, bg_color, fg_color, border_color, background_corner_colors, overwrite_preferred_drawing_method, **kwargs)

        self.play_button = ctk.CTkButton(master=self).grid(row=0, column=0, padx=(8, 4), pady=(8, 8))
        
    
class VideoCard(ctk.CTkFrame):
    def __init__(self, master: Any, width: int = 200, height: int = 200, corner_radius: int | str | None = None, border_width: int | str | None = None, bg_color: str | Tuple[str] = "transparent", fg_color: str | Tuple[str] | None = None, border_color: str | Tuple[str] | None = None, background_corner_colors: Tuple[str | Tuple[str]] | None = None, overwrite_preferred_drawing_method: str | None = None, **kwargs):
        super().__init__(master, width, height, corner_radius, border_width, bg_color, fg_color, border_color, background_corner_colors, overwrite_preferred_drawing_method, **kwargs)
        
        self.label_2 = ctk.CTkLabel(master=self, text='Label 2').grid(row=0, column=0, padx=8, pady=8)



        


if __name__ == '__main__':
    ctk.set_appearance_mode('system')
    ctk.set_default_color_theme('dark-blue')

    root = MainWindow()
    root.mainloop()