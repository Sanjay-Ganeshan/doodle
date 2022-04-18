from cmath import pi
import pygame
import threading
import time
import typing as T
import os
import numpy as np
from matplotlib.colors import hsv_to_rgb
from colorsys import hsv_to_rgb as single_hsv_to_rgb
from PIL import Image as PILImage

pygame.init()

def keep_aspect_ratio(original_size: T.Tuple[int, int], new_size_max: T.Tuple[int, int]) -> T.Tuple[T.Tuple[int, int], int, int]:
    """
    Returns ((new width, new height), X offset, Y offset)
    """
    w0, h0 = original_size
    w1, h1 = new_size_max

    sW = w1 / w0
    sH = h1 / h0

    scale = min(sW, sH)

    w2, h2 = int(w0 * scale), int(h0 * scale)
    assert w2 <= w1 and h2 <= h1

    w_margin = (w1 - w2) // 2
    h_margin = (h1 - h2) // 2

    return ((w2, h2), w_margin, h_margin)

def tinted(texture: pygame.Surface, color: pygame.Color):
    cp = texture.copy()
    cp.fill(color, None, pygame.BLEND_MULT)
    return cp

def make_color_wheel(size: int, v: float) -> pygame.Surface:
    v = np.clip(v, 0.0, 1.0)
    size = max(5, (size // 2) * 2 - 1)
    semi_size = size // 2
    # Each row is same
    x_vals = np.broadcast_to(np.arange(-semi_size, semi_size+1, 1).reshape((-1, 1)), (size, size)).reshape((size*size,))

    # Each column is same
    y_vals = np.broadcast_to(np.arange(-semi_size, semi_size+1, 1).reshape((1, -1)), (size, size)).reshape((size*size,))

    r_vals = np.hypot(x_vals, y_vals) / semi_size
    theta_vals = (np.arctan2(y_vals, x_vals) + np.pi) / (2*np.pi)
    v_vals = np.full_like(r_vals, v)
    #a_vals = np.ones_like(r_vals)
    v_vals[r_vals > 1] = 0

    # (3, w*h)
    hsv_flat = np.array([theta_vals, r_vals, v_vals])
    hsv_shaped = hsv_flat.reshape((3, size, size)).transpose((1, 2, 0))
    #a_shaped = a_vals.reshape((1, size, size)).transpose((1, 2, 0))

    rgb = hsv_to_rgb(np.clip(hsv_shaped, 0, 1))
    #rgba = np.concatenate([rgb, a_shaped], axis=2)
    
    surf = pygame.Surface((size, size))
    rgb_as_img_arr = np.clip(rgb*255, 0, 255).astype(np.uint8)
    pygame.surfarray.blit_array(surf, rgb_as_img_arr)
    surf.set_colorkey(pygame.Color(0, 0, 0))
    
    return surf
    








class Layer:
    def __init__(self, canvas: "Canvas", name: str):
        self.canvas: "Canvas" = canvas
        self.name: str = name
        self.width: int = self.canvas.width
        self.height: int = self.canvas.height

        self.contents = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.contents.fill(pygame.Color(0, 0, 0, 0))
    
    def stamp(self, texture: pygame.Surface, center: T.Tuple[int, int], blend: bool) -> None:
        w, h = texture.get_size()
        left = center[0] - (w // 2)
        top = center[1] - (h // 2)

        out_top = max(0, top)
        out_left = max(0, left)

        src_left = max(0, -1 * left)
        src_top = max(0, -1 * top)
        src_w = w - src_left
        src_h = h - src_top
        src_rect = pygame.Rect(
            max(0, -1 * left),
            max(0, -1 * top),
            src_w,
            src_h
        )

        self.contents.blit(texture, (out_left, out_top), src_rect, special_flags=(pygame.BLEND_ALPHA_SDL2 if blend else pygame.BLEND_RGBA_MIN))
    
    def stroke(self, texture: pygame.Surface, start: T.Tuple[int, int], finish: T.Tuple[int, int], blend: bool) -> None:
        x0, y0 = start
        x1, y1 = finish
        distance = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        tex_width = texture.get_width()

        # Want multiple stamps per texture width
        stamp_every = tex_width / 4
        n_stamps = int(distance / stamp_every) + 1
        dx = (x1 - x0) / n_stamps
        dy = (y1 - y0) / n_stamps
        for i in range(n_stamps):
            x = int(x0 + dx * i)
            y = int(y0 + dy * i)
            self.stamp(texture, (x,y), blend=blend)
        
        

def relative_pos(position, area_size, area_left, area_top, src_size) -> T.Optional[T.Tuple[int, int]]:
    area_right = area_left + area_size[0]
    area_bottom = area_top + area_size[1]
    x, y = position
    if x < area_left or x >= area_right or y < area_top or y >= area_bottom:
        return None
    else:
        norm_x = (x - area_left) / area_size[0]
        norm_y = (y - area_top) / area_size[1]
        
        scaled_x = int(norm_x * src_size[0])
        scaled_y = int(norm_y * src_size[1])
        return (scaled_x, scaled_y)

class Brush:
    def __init__(self, size: int, texture: pygame.Surface, tint: T.Optional[pygame.Color], is_eraser: bool):
        self.size: int = size
        self.texture: pygame.Surface = texture
        self.tint: T.Optional[pygame.Color] = tint
        self.eraser = is_eraser
    
    def create(self) -> pygame.Surface:
        scaled = pygame.transform.smoothscale(self.texture, (self.size, self.size))
        if self.eraser:
            scaled.set_alpha(0)
            return scaled
        else:
            if self.tint is None:
                return scaled
            else:
                return tinted(scaled, self.tint)

def load_stamps() -> T.List[pygame.Surface]:
    mydir = os.path.dirname(os.path.abspath(__file__))
    stamps = os.path.join(mydir, "stamps")
    stamp_files = sorted([fn for fn in os.listdir(stamps) if os.path.splitext(fn)[1]==".png"])
    loaded = []
    for fn in stamp_files:
        abs_to_file = os.path.join(stamps, fn)
        loaded.append(pygame.image.load(abs_to_file))
    return loaded

def get_save_path() -> str:
    mydir = os.path.dirname(os.path.abspath(__file__))
    save_to = None
    n = 0
    while save_to is None or os.path.exists(save_to):
        save_to = os.path.join(mydir, "saved", f"img{n}.png")
        n += 1
    return save_to

class BrushSettings:
    def __init__(self):
        self.one_and_done: bool = True
        
        self.stroke_prev_position = None
        self.erasing = False

        self.cached_texture: T.Optional[pygame.Surface] = None

        self.color: T.Optional[pygame.Color] = None
        self.size: int = 25
        self.textures = load_stamps()
        self.wheel_v: float = 1.0

        assert len(self.textures) > 0, "No stamps"
        self.selection = 0
    
    def make_brush(self) -> Brush:
        return Brush(self.size, self.textures[self.selection], self.color, self.erasing)      

    def get_texture(self) -> pygame.Surface:
        if self.cached_texture is None:
            self.cached_texture = self.make_brush().create()
        return self.cached_texture

    def bigger(self) -> None:
        if self.stroke_prev_position:
            return

        self.size += 2
        self.cached_texture = None

    def smaller(self) -> None:
        if self.stroke_prev_position:
            return

        if self.size >= 3:
            self.size -= 2
            self.cached_texture = None
    
    def next_brush(self) -> None:
        if self.stroke_prev_position:
            return

        self.selection = (self.selection + 1 ) % len(self.textures)
        self.cached_texture = None
    
    def prev_brush(self) -> None:
        if self.stroke_prev_position:
            return

        self.selection = ((self.selection - 1) + len(self.textures)) % len(self.textures)
        self.cached_texture = None
    
    def choose_brush(self, which: int) -> None:
        if self.stroke_prev_position:
            return
        
        if which >= 0 and which <= len(self.textures):
            self.selection = which
            self.cached_texture = None

    def set_color(self, color: pygame.Color):
        if self.stroke_prev_position:
            return

        self.color = color
        self.cached_texture = None
    
    def clear_color(self):
        if self.stroke_prev_position:
            return

        self.color = None
        self.cached_texture = None

    def toggle_mode(self):
        if self.stroke_prev_position:
            return
        
        self.one_and_done = not self.one_and_done
    
    def toggle_eraser(self):
        if self.stroke_prev_position:
            return
        
        self.erasing = not self.erasing
        if self.erasing:
            self.choose_brush(0)
        self.cached_texture = None

class Canvas:
    def __init__(self, width=640, height=480):
        self.should_quit: threading.Event = threading.Event()
        self.width: int = width
        self.height: int = height
        self.layers: T.List[Layer] = [Layer(self, "Background")]
        self.selected_layer = 0
        
        self.brush_settings = BrushSettings()

        self.mouse_pos = pygame.mouse.get_pos()
        self.win_size = (self.width, self.height)
        self.prev_win_size = self.win_size

        self.color_wheel: T.Optional[pygame.Surface] = None

        self.lock_x = None
        self.lock_y = None

        self.save_name = None

    def save(self):
        img = self.draw_main_area()
        arr = pygame.surfarray.array3d(img)
        pilim = PILImage.fromarray(arr.transpose((1, 0, 2)))
        
        if self.save_name is None:
            self.save_name = get_save_path()
        
        pilim.save(self.save_name)
        print("Saved to", self.save_name)
        
        

    def get_mouse_rel_main_area(self) -> T.Optional[T.Tuple[int, int]]:
        mouse_rel_main_area = relative_pos(
            self.mouse_pos, *self.get_main_area_bounds(), (self.width, self.height)
        )
        return mouse_rel_main_area

    def get_mouse_rel_color_wheel(self) -> T.Optional[T.Tuple[int, int]]:
        mouse_rel_color_wheel = relative_pos(
            self.mouse_pos, *self.get_color_wheel_bounds(), (1000, 1000)
        )
        return mouse_rel_color_wheel

    def d_add_layer(self) -> int:
        self.layers.append(Layer(self, "Layer"))
        return len(self.layers) - 1
    
    def d_delete_current_layer(self) -> int:
        assert len(self.layers) > 0
        if len(self.layers) == 1:
            return 0
        print("Delete Layer")
        # we're good
        self.layers.pop(self.selected_layer)
        assert len(self.layers) > 0
        while self.selected_layer >= len(self.layers):
            self.selected_layer -= 1
        return self.selected_layer

    def d_select_layer(self, layer_id: int) -> bool:
        if layer_id >= 0 and layer_id < len(self.layers):
            print(f"Now on {self.layers[layer_id].name}")
            self.selected_layer = layer_id
            return True
        else:
            return False

    def update(self):
        first_event = pygame.event.wait()
        self.mouse_pos = pygame.mouse.get_pos()
        self.win_size = pygame.display.get_window_size()

        if self.win_size != self.prev_win_size:
            self.color_wheel = None
            self.prev_win_size = self.win_size

        self.lock_mouse_axes()
        self.handle_all_events(first_event)
        self.handle_mouse_move()
    
    def draw_main_area(self) -> pygame.Surface:
        buffer = pygame.Surface((self.width, self.height))
        buffer.fill(pygame.Color(255, 255, 255, 255))
        for each_layer in self.layers:
            buffer.blit(each_layer.contents, (0,0), None, pygame.BLEND_ALPHA_SDL2)
        rel_main_area = self.get_mouse_rel_main_area()
        if rel_main_area is not None:
            brush_tex = self.brush_settings.get_texture()
            mx, my = rel_main_area
            buffer.blit(brush_tex, (mx - self.brush_settings.size // 2, my - self.brush_settings.size // 2), None, pygame.BLEND_ALPHA_SDL2)
        return buffer

    def draw_color_wheel(self) -> pygame.Surface:
        if self.color_wheel is None:
            scaled_size, x_offset, y_offset = self.get_color_wheel_bounds()
            self.color_wheel = make_color_wheel(scaled_size[0], self.brush_settings.wheel_v)
        return self.color_wheel

    def get_main_area_bounds(self) -> T.Tuple[T.Tuple[int, int], int, int]:
        (win_w, win_h) = self.win_size
        scaled_size, x_offset, y_offset = keep_aspect_ratio((self.width, self.height), (int(win_w * 0.8), win_h))
        return scaled_size, x_offset, y_offset
    
    def get_color_wheel_bounds(self) -> T.Tuple[T.Tuple[int, int], int, int]:
        (win_w, win_h) = self.win_size
        scaled_size, x_offset, y_offset = keep_aspect_ratio(
            (100, 100), (int(win_w * 0.2), int(win_h * 0.3)))
        return scaled_size, int(x_offset + (win_w * 0.8)), y_offset

    

    def draw(self):
        # 
        # Populate the image buffer
        # Draw all layers
        main_area = self.draw_main_area()
        wheel = self.draw_color_wheel()
        
        display = pygame.display.get_surface()

        # Clear the display
        display.fill(pygame.color.Color(0, 0, 0, 255))

        # The left 80% is reserved for the canvas
        scaled_size, canv_x, canv_y = self.get_main_area_bounds()
        # We want to keep the aspect ratio the same
        wheel_scaled_size, wheel_x, wheel_y = self.get_color_wheel_bounds()
        
        canv_draw = pygame.transform.smoothscale(main_area, scaled_size)
        wheel_draw = pygame.transform.smoothscale(wheel, wheel_scaled_size)

        display.blit(canv_draw, (canv_x, canv_y))
        display.blit(wheel_draw, (wheel_x, wheel_y))
        
        # Draw the image canvas
        pygame.display.flip()


    def handle_left_click_main_area(self, pos: T.Tuple[int, int]) -> None:
        self.layers[self.selected_layer].stamp(self.brush_settings.get_texture(), pos, blend=not self.brush_settings.erasing)
        if not self.brush_settings.one_and_done:
            self.brush_settings.stroke_prev_position = pos
    
    def handle_mouse_move(self) -> None:
        rel_main_area = self.get_mouse_rel_main_area()
        if self.brush_settings.stroke_prev_position and rel_main_area is not None:
            self.layers[self.selected_layer].stroke(self.brush_settings.get_texture(), self.brush_settings.stroke_prev_position, rel_main_area, blend=not self.brush_settings.erasing)
            self.brush_settings.stroke_prev_position = rel_main_area
    
    def handle_left_click_color_wheel(self, pos: T.Tuple[int, int]) -> None:
        normx = (pos[0] - 500) / 500
        normy = (pos[1] - 500) / 500
        r = np.hypot(normx, normy)
        if r <= 1.0:
            theta = (np.arctan2(normy, normx) + np.pi) / (2 * np.pi)
            v = self.brush_settings.wheel_v
            h = theta
            s = r
            cr, cg, cb = single_hsv_to_rgb(h,s,v)
            color = pygame.Color(int(cr*255), int(cg*255), int(cb*255))
            self.brush_settings.set_color(color)

    def handle_right_click_color_wheel(self, pos: T.Tuple[int, int]) -> None:
        normx = pos[0] / 1000
        normy = pos[1] / 1000
        r = np.hypot(normx, normy)
        if r <= 1.0:
            self.brush_settings.clear_color()


    def lock_mouse_axes(self) -> None:
        current_pos = self.mouse_pos
        new_pos = list(current_pos)
        if self.lock_x is not None:
            new_pos[0] = self.lock_x
        if self.lock_y is not None:
            new_pos[1] = self.lock_y
        new_pos = tuple(new_pos)
        if new_pos != current_pos:
            pygame.mouse.set_pos(new_pos)
            self.mouse_pos = new_pos
        
        


    def handle_all_events(self, first_event: pygame.event):
        events = [first_event] + pygame.event.get()
        
        mouse_rel_main = self.get_mouse_rel_main_area()
        mouse_rel_wheel = self.get_mouse_rel_color_wheel()

        if mouse_rel_main is None:
            pygame.mouse.set_visible(True)
        else:
            pygame.mouse.set_visible(self.brush_settings.erasing)

        for ev in events:
            if ev.type == pygame.QUIT:
                self.should_quit.set()
            elif ev.type == pygame.KEYDOWN:
                key = ev.key # What key was pressed this frame
                mod = ev.mod # Bitmask of key modifiers
                if key == pygame.K_ESCAPE:
                    self.should_quit.set()
                if not self.brush_settings.stroke_prev_position and (mod & pygame.KMOD_CTRL):
                    if key == pygame.K_n:
                        print("New Layer")
                        new_layer = self.d_add_layer()
                        self.d_select_layer(new_layer)
                    if key == pygame.K_d:
                        self.d_delete_current_layer()
                    if key == pygame.K_UP:
                        self.d_select_layer(self.selected_layer + 1)
                    if key == pygame.K_DOWN:
                        self.d_select_layer(self.selected_layer - 1)
                    if key == pygame.K_s:
                        if mod & pygame.KMOD_SHIFT:
                            self.save_name = None
                        self.save()
                else:
                    if key == pygame.K_LEFTBRACKET:
                        self.brush_settings.smaller()
                    if key == pygame.K_RIGHTBRACKET:
                        self.brush_settings.bigger()
                    if key == pygame.K_o:
                        self.brush_settings.prev_brush()
                    if key == pygame.K_p:
                        self.brush_settings.next_brush()
                    if key == pygame.K_e:
                        self.brush_settings.toggle_eraser()
                    if key == pygame.K_y:
                        self.lock_x = self.mouse_pos[0]
                    if key == pygame.K_x:
                        self.lock_y = self.mouse_pos[1]
                    if key == pygame.K_b:
                        self.brush_settings.toggle_mode()
            elif ev.type == pygame.KEYUP:
                key = ev.key
                mod = ev.mod
                if key == pygame.K_y:
                    self.lock_x = None
                if key == pygame.K_x:
                    self.lock_y = None
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                button = ev.button

                if button == pygame.BUTTON_LEFT:
                    if mouse_rel_main is not None:
                        self.handle_left_click_main_area(mouse_rel_main)
                    if mouse_rel_wheel is not None:
                        self.handle_left_click_color_wheel(mouse_rel_wheel)
                if button == pygame.BUTTON_RIGHT:
                    if mouse_rel_wheel is not None:
                        self.handle_right_click_color_wheel(mouse_rel_wheel)
                
            elif ev.type == pygame.MOUSEBUTTONUP:
                button = ev.button

                if button == pygame.BUTTON_LEFT:
                    if self.brush_settings.stroke_prev_position:
                        self.brush_settings.stroke_prev_position = None

def main():
    canvas = Canvas()
    pygame.display.set_mode((800, 800), flags=(pygame.RESIZABLE))
    fps = 30
    spf = 1 / fps
    while not canvas.should_quit.is_set():
        t0 = time.time()
        canvas.update()
        canvas.draw()
        t1 = time.time()
        elapsed = t1 - t0



if __name__ == "__main__":
    main()

