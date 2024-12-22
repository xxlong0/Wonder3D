from copy import deepcopy
from re import S
from typing import Union
import torch
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from core.opt import MeshOptimizer
import numpy as np
from util.func import to_numpy

from util.snapshot import Snapshot
    

class Viewer:
    def __init__(self, 
            target_vertices:torch.Tensor, #V,3 
            target_faces:torch.Tensor, #F,3 
            snapshots:list[Snapshot],
            vertex_colors:dict[str,list[np.array]]
            ):
        self._target_vertices = target_vertices
        self._target_faces = target_faces
        self._snapshots = snapshots
        self._vertex_colors = vertex_colors

        self._window_o3 = gui.Application.instance.create_window("Continuous Remeshing",1000,800)
        self._window_o3.set_on_layout(self._layout)
        self._scene_widget = gui.SceneWidget()
        self._scene_widget.scene = rendering.Open3DScene(self._window_o3.renderer)
        self._scene_widget.scene.set_background([.5, .5, .5, 1])
        bbox = o3d.geometry.AxisAlignedBoundingBox([-1, -1, -1], [1, 1, 1])
        self._scene_widget.setup_camera(60, bbox, [0, 0, 0])
        self._window_o3.add_child(self._scene_widget)
        self._scene_widget.set_on_mouse(self._on_mouse)

        #lights
        self._scene_widget.scene.scene.enable_sun_light(False)            
        self._scene_widget.scene.scene.set_indirect_light(gui.Application.instance.resource_path + '/park2')
        self._scene_widget.scene.scene.enable_indirect_light(True)
        self._scene_widget.scene.scene.set_indirect_light_intensity(45000)
        self._scene_widget.scene.show_skybox(False)

        #right panel
        margins = gui.Margins(*[self._window_o3.theme.default_margin]*4)
        spacing = self._window_o3.theme.default_layout_spacing
        self._right_panel = gui.Vert(spacing,margins)
        
        def make_checkbox(name,checked):
            checkbox = gui.Checkbox(name)
            checkbox.checked = checked
            checkbox.set_on_checked(lambda *args:self._update())
            self._right_panel.add_child(checkbox)
            return checkbox

        self._mesh_checkbox = make_checkbox("Show Mesh",True)
        self._colorbox = gui.Combobox()
        for item in ['Gray','Relative Velocity nu','Reference Edge Length l_ref',*self._vertex_colors.keys()]:
            self._colorbox.add_item(item)
        self._colorbox.set_on_selection_changed(lambda *args:self._update())
        self._right_panel.add_child(self._colorbox)
        
        self._clim_slider = gui.Slider(gui.Slider.DOUBLE)
        self._clim_slider.double_value = .2
        self._clim_slider.set_limits(1e-3, 1)
        self._clim_slider.set_on_value_changed(lambda *args:self._update())
        self._right_panel.add_child(self._clim_slider)

        self._edges_checkbox = make_checkbox("Show Edges",True)
        self._target_mesh_checkbox = make_checkbox("Show Target Mesh",False)
        self._right_panel.add_child(gui.Label('Ctrl-Click Mesh For Plot!'))
        self._target_edges_checkbox = make_checkbox("Show Target Edges",False)
        self._positions_checkbox = make_checkbox("Plot Positions",False)
        self._gradients_checkbox = make_checkbox("Plot Gradients",False)
        self._m1_checkbox = make_checkbox("Plot m1",False)
        self._m2_checkbox = make_checkbox("Plot m2",False)
        self._nu_checkbox = make_checkbox("Plot nu",True)
        self._lref_checkbox = make_checkbox("Plot l_ref",True)
        self._window_o3.add_child(self._right_panel)

        #bottom panel
        self._bottom_panel = gui.VGrid(cols=2,spacing=spacing)
        self._snapshot_slider = gui.Slider(gui.Slider.INT)
        self._snapshot_slider.int_value = len(self._snapshots)-1
        self._snapshot_slider.set_limits(0, len(self._snapshots)-1)
        self._snapshot_slider.set_on_value_changed(lambda *args:self._update())
        self._bottom_panel.add_child(self._snapshot_slider)
        self._window_o3.add_child(self._bottom_panel)

        self._update()
        
    def _update(self):
        snapshot = self._snapshots[self._snapshot_slider.int_value]
        
        self._scene_widget.scene.clear_geometry()

        self._scene_widget.scene.show_axes(True)
        
        MaterialType = rendering.MaterialRecord if hasattr(rendering,'MaterialRecord') else rendering.Material 

        def add_mesh(name,color,vertices,faces,show_mesh,show_edges,vertex_colors=None):
            vertices_np = vertices.detach().cpu().numpy()
            vertices_o3 = o3d.utility.Vector3dVector(vertices_np)
            faces_o3 = o3d.utility.Vector3iVector(faces.type(torch.int32).cpu().numpy())
            triangleMesh = o3d.geometry.TriangleMesh(vertices_o3,faces_o3)
            triangleMesh.compute_vertex_normals()
            if vertex_colors is not None:
                vertex_colors_np = to_numpy(vertex_colors)
                triangleMesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors_np)

            if show_mesh:
                material = MaterialType()
                if vertex_colors is None:
                    material.shader = "defaultLit"
                    material.base_color = color
                self._scene_widget.scene.add_geometry(f"{name}_triangleMesh", triangleMesh, material)

            if show_edges:
                edges_material = MaterialType()
                edges_material.base_color = [0,0,0,1]
                edges_material.shader = "unlitLine"
                edges_lineset = o3d.geometry.LineSet.create_from_triangle_mesh(triangleMesh)
                edges_lineset.points = o3d.utility.Vector3dVector(vertices_np + 1e-4 * np.asarray(triangleMesh.vertex_normals))
                self._scene_widget.scene.add_geometry(f"{name}_edges_lineset", edges_lineset, edges_material)

        clim = self._clim_slider.double_value
        if self._colorbox.selected_text=='Relative Velocity nu' and isinstance(snapshot.optimizer, MeshOptimizer):
            vertex_colors = snapshot.optimizer._nu
        elif self._colorbox.selected_text=='Reference Edge Length l_ref' and isinstance(snapshot.optimizer, MeshOptimizer):
            vertex_colors = snapshot.optimizer._ref_len
        elif self._colorbox.selected_text in self._vertex_colors.keys():
            vertex_colors = self._vertex_colors[self._colorbox.selected_text][self._snapshot_slider.int_value]
        else:
            vertex_colors = None

        if vertex_colors is not None:
            c = (to_numpy(vertex_colors) / clim).clip(0,1)
            vertex_colors = np.stack((c,1-c,np.zeros_like(c)),axis=-1)

        add_mesh("mesh",[.5,.5,.5,1],snapshot.vertices,snapshot.faces,self._mesh_checkbox.checked,self._edges_checkbox.checked, vertex_colors)
        add_mesh("target",[.5,.5,1,1],self._target_vertices,self._target_faces,self._target_mesh_checkbox.checked,self._target_edges_checkbox.checked)

    def _layout(self,layout_context):
        r = self._window_o3.content_rect
        
        h = self._bottom_panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height
        self._bottom_panel.frame = gui.Rect(0, r.height - h, r.width, h)
        r.height -= h

        w = 250
        self._right_panel.frame = gui.Rect(r.width - w, 0, w, r.height)
        r.width -= w

        self._scene_widget.frame = r

    def _on_mouse(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.CTRL):
            self._hit_test(event)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED
        
    def _hit_test(self,event):
        def depth_callback(depth_image):
            f = self._scene_widget.frame
            depth = np.asarray(depth_image)[event.y - f.y, event.x - f.x]
            if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                return
            #need to flip y https://github.com/isl-org/Open3D/issues/4244
            pos = self._scene_widget.scene.camera.unproject(event.x - f.x, f.height - event.y, depth, f.width, f.height)
            
            opt = self._snapshots[self._snapshot_slider.int_value].optimizer
            vertex = (opt.vertices.cpu() - torch.tensor(pos)).norm(dim=-1).argmin().item()
            
            gui.Application.instance.post_to_main_thread(self._window_o3, lambda:self._on_click(pos,vertex))

        self._scene_widget.scene.scene.render_to_depth_image(depth_callback)
        
    def _on_click(self,pos,vertex):
        self._show_plot(vertex)

    def _show_plot(self,vertex):
        ind = self._snapshot_slider.int_value
        device = self._snapshots[0].vertices.device
        vert_ind = torch.zeros(len(self._snapshots),dtype=torch.long,device=device)
        vert_ind[ind] = vertex

        def trace(i):
            nonlocal cur_pos
            vert_ind[i] = (self._snapshots[i].vertices - cur_pos).norm(dim=-1).argmin(dim=0)
            cur_pos = self._snapshots[i].vertices[vert_ind[i]]

        cur_pos = self._snapshots[ind].vertices[vertex] 
        for i in range(ind-1,-1,-1):
            trace(i)

        cur_pos = self._snapshots[ind].vertices[vertex] 
        for i in range(ind+1,len(self._snapshots)):
            trace(i)   

        dims = slice(None,None)

        grad_scale = 100

        from cycler import cycler
        import matplotlib.pyplot as plt
        plt.gca().set_prop_cycle(cycler(linestyle=['-', '--', ':'][dims]))
        
        def extract(prop):
            values = [prop(self._snapshots[i].optimizer,vert_ind[i]) for i in range(0,len(vert_ind))]
            if isinstance(values[0],torch.Tensor):
                values = torch.stack(values).cpu() 
            return values
            
        s = [s.optimizer._step for s in self._snapshots]

        if self._positions_checkbox.checked:
            plt.plot(s,extract(lambda opt,v:opt.vertices[v,dims]),'b',label='pos')
        if self._gradients_checkbox.checked:
            plt.plot(s,grad_scale*extract(lambda opt,v:opt.vertices.grad[v,dims]),'k',label='grad')

        m1 = extract(lambda opt,v:opt._m1[v])
        m2 = extract(lambda opt,v:opt._m2[v])
        velocity = m1 / m2[:,None].sqrt().add_(1e-8) #V,3
        speed = velocity.norm(dim=-1)
        if self._m1_checkbox.checked:
            plt.plot(s,grad_scale*extract(lambda opt,v:opt._m1[v,dims]),'r',label='m1')
        if self._m2_checkbox.checked:
            plt.plot(s,grad_scale*extract(lambda opt,v:opt._m2[v].sqrt()),'-m',label='m2.sqrt()')
        if self._nu_checkbox.checked:
            plt.plot(s,speed,color='orange',label='speed')
            plt.plot(s,extract(lambda opt,v:opt._nu[v]),'-c',label='nu')
        if self._lref_checkbox.checked:
            plt.plot(s,extract(lambda opt,v:opt._ref_len[v]),color='gray',label='l_ref')

        plt.axvline(x=ind, color='k')
        plt.legend()
        plt.grid()
        plt.show()


def show(
    target_vertices:torch.Tensor, #V,3 
    target_faces:torch.Tensor, #F,3 
    snapshots:list[Snapshot],
    vertex_colors:dict[str,list[np.array]]={}
    ):
    for vc in vertex_colors.values():
        assert [c.shape[0] for c in vc] == [s.vertices.shape[0] for s in snapshots]

    gui.Application.instance.initialize()
    viewer = Viewer(target_vertices,target_faces,snapshots,vertex_colors)
    gui.Application.instance.run()