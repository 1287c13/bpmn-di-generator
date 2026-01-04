import xml.etree.ElementTree as ElementTree
from typing import List, Literal, Dict, Set, Tuple, Optional
from functools import reduce
import tkinter as tk
from tkinter import filedialog
import re
import os.path
from tkinter import ttk


class BpmnXmlManager:
  def __init__(self, input_xml_path='diagram.bpmn'):
    self.namespaces = {
      'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
      'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI',
      'dc': 'http://www.omg.org/spec/DD/20100524/DC',
      'di': 'http://www.omg.org/spec/DD/20100524/DI'}
    self.root_tag_name = 'bpmn:process'
    self.input_xml_path = input_xml_path
    self.input_xml = None
    self.root = None
    self.sizes_dict = {}

    if self.input_xml_path:
      try:
        with open(self.input_xml_path, 'r') as file:
          self.input_xml = file.read()
      except FileNotFoundError:
        pass

  def set_input_xml(self, input_xml):
    self.input_xml = input_xml
    self.root = ElementTree.ElementTree(
      ElementTree.fromstring(self.input_xml)).getroot()

  @staticmethod
  def parse_element(element: ElementTree.Element):
    """
    Получает элемент BPMN схемы.
    Возвращает словарь в котором ключ это идентификатор элемента, а значение
    это словарь всех атрибутов элемента.
    """
    attrs = {
      'tag': element.tag.split('}')[-1],
      'text': '' if not element.text else element.text.strip()
    }

    if element.attrib:
      attrs.update(element.attrib)

    children = list(element)
    if len(children):
      child_elements = {}
      for child in children:
        child_elements.update(BpmnXmlManager.parse_element(child))
      attrs['children'] = child_elements

    try:
      return {attrs['id']: attrs}
    except KeyError:
      return {attrs['text']: attrs}

  def extract_process_dict_repr(self):
    root = ElementTree.fromstring(self.input_xml)
    processes = root.findall(
      f'./{self.root_tag_name}', namespaces=self.namespaces)

    if not processes:
      raise ValueError("BPMN XML is not valid: aint no <bpmn:process> tag")

    # todo добавить поддержку схем с несколькими процессами
    first_process = processes[0]
    data = BpmnXmlManager.parse_element(first_process)

    return data

  def generate_di_layer_xml(self, di_layer_dict):
    for k, v in self.namespaces.items():
      ElementTree.register_namespace(k, v)

    root = ElementTree.fromstring(self.input_xml)
    shapes = root.findall(".//bpmndi:BPMNShape", namespaces=self.namespaces)
    for shape in shapes:
      element_id = shape.attrib['bpmnElement']
      if element_id in di_layer_dict:
        bounds = shape.find("./dc:Bounds", namespaces=self.namespaces)
        bounds.attrib['x'] = str(di_layer_dict[element_id]['x'])
        bounds.attrib['y'] = str(di_layer_dict[element_id]['y'])

        if ('isExpanded' in shape.attrib
            and shape.attrib['isExpanded'] == 'true') \
           or ('spec' in di_layer_dict[element_id]
            and di_layer_dict[element_id]['spec'] == 'lane'):

          bounds.attrib['width'] = str(di_layer_dict[element_id]['w'])
          bounds.attrib['height'] = str(di_layer_dict[element_id]['h'])

    return '<?xml version="1.0" encoding="UTF-8"?>\n' + \
           ElementTree.tostring(root, encoding='unicode', method='xml').strip()

  def collect_sizes(self):
    self.sizes_dict = {}

    root = ElementTree.fromstring(self.input_xml)
    diagram_node = root.find(
      './/bpmndi:BPMNDiagram', namespaces=self.namespaces)

    if diagram_node is not None:
      shapes = diagram_node.findall(
        './/bpmndi:BPMNShape', namespaces=self.namespaces)
      for shape in shapes:
        bounds = shape.find('dc:Bounds', namespaces=self.namespaces)
        if bounds is not None:
          element_id = shape.get('bpmnElement')
          width = float(bounds.get('width'))
          height = float(bounds.get('height'))
          self.sizes_dict[element_id] = {'width': width, 'height': height}


class Subprocess(dict):
  def __init__(self, data: Dict):
    super().__init__()
    self.update(data)
    self.id: Optional[str]
    self.lane: Optional[int]
    self.grid: Dict[str, Dict[str, List[int]]] = {}


class BpmnLayoutGenerator:

  def __init__(self):
    self.branch_counter: int = 1
    self.repr: dict = {}
    self.sizes: dict = {}
    self.grid: Dict[Literal['cols', 'rows'], List[float]] = {}
    self.subprocesses: list = []
    self.start_events_ids: List[str] = []
    self.visited_nodes_ids: List[str] = []
    self.nodes_to_visit_ids: List[str] = []
    self.num_of_brunches: int = 0
    self.visual_indent: float = 12.0
    self.elem_params: Dict[str, Dict[
      Literal['id', 'c', 'r', 'w', 'h', 'x', 'y', 'spec'],
      str or int or float
    ]] = {}

  def generate_di_layer(self, tags_dict_repr, sizes):
    self.repr = list(tags_dict_repr.values())[0]['children']
    self.sizes = sizes

    self.call_process_handler('add_structure_attrs')
    self.call_process_handler('calc_grid_structure')
    self.calc_grid_sizes()
    self.call_process_handler('calc_elems_coords')
    self.update_pool()
    self.calc_edges()

  @staticmethod
  def _get_arrow_endpoint_node(
          arrow_id, direction: Literal['source', 'target'], structure):

    return structure[arrow_id][f'{direction}Ref']

  @staticmethod
  def _get_connected_flows_keyvalues(
          parent_node_id,
          structure,
          direction: Literal['incoming', 'outgoing']):

    return filter(
      lambda x: x[1]['tag'] == direction,
      structure[parent_node_id]['children'].items())

  def _get_connected_nodes_ids(
          self,
          parent_node_id,
          structure,
          direction: Literal['source', 'target']):

    flow_direction: Literal['incoming', 'outgoing'] = 'incoming'
    if direction == 'target':
      flow_direction = 'outgoing'

    connected_flows_keyvalues = self._get_connected_flows_keyvalues(
      parent_node_id, structure, flow_direction)

    return list(map(
      lambda x: self._get_arrow_endpoint_node(x[0], direction, structure),
      connected_flows_keyvalues))

  def _explore_neighboring_nodes(self, parent_node_id, structure):
    """
    Ищем следующие элементы. Первый возвращаем,
    остальные помечаем к посещению.
    """
    target_nodes_ids = self._get_connected_nodes_ids(
      parent_node_id, structure, 'target')

    self.nodes_to_visit_ids += target_nodes_ids[1:]

    try:
      structure[target_nodes_ids[0]].setdefault('branch', self.branch_counter)
      return target_nodes_ids[0]
    except IndexError:
      return None

  def _add_subprocess(self, structure, elem_id):
    self.subprocesses.append(
      Subprocess(structure[elem_id]['children']))
    self.subprocesses[-1].id = elem_id
    self.subprocesses[-1].lane = self._get_elem_lane_number(elem_id) \
                                 or structure.lane

  def _traverse_and_assign_branch_numbers(self, initial_elem_id, structure):
    """
    Проходим всю ветку начального элемента и делаем рекурсивный вызов
    для следующих обнаруженных элементов, которые нужно посетить.
    """

    if 'branch' not in structure[initial_elem_id]:

      structure[initial_elem_id]['branch'] = self.branch_counter
      if structure[initial_elem_id]['tag'] == 'subProcess':
        self._add_subprocess(structure, initial_elem_id)

      next_elem_id = self._explore_neighboring_nodes(initial_elem_id,
                                                     structure)
      while next_elem_id:
        if structure[next_elem_id]['tag'] == 'subProcess':
          self._add_subprocess(structure, next_elem_id)

        next_elem_id = self._explore_neighboring_nodes(next_elem_id, structure)

      self.branch_counter += 1

    next_branch_first_elem = self.nodes_to_visit_ids.pop()
    try:
      self._traverse_and_assign_branch_numbers(next_branch_first_elem,
                                               structure)
    except IndexError:
      pass

  @staticmethod
  def _get_start_events_ids(process):
    return [v['id'] for k, v in process.items() if
            v.get('tag') == 'startEvent']

  def add_structure_attrs(self, structure):
    """
    Собираем все стартовые события и запускаем разметку ветвей схемы.
    """
    self.branch_counter = 1
    self.start_events_ids = self._get_start_events_ids(structure)
    for initial_elem_id in self.start_events_ids:
      self._traverse_and_assign_branch_numbers(initial_elem_id, structure)

  def call_process_handler(self, handler_name):
    getattr(self, handler_name)(self.repr)

    for subprocess in self.subprocesses:
      getattr(self, handler_name)(subprocess)

  def calc_grid_structure(self, structure):
    """
    Итерируемся по номерам столбцов и подбираем элементы для размещения в них.
    todo рефачить
    """
    current_col_elems_ids: List[str] = self._get_start_events_ids(structure)
    delayed_processing_queue: Set[str] = set([])
    source_nodes_ids_cache: Dict[str: Tuple[str]] = {}
    target_nodes_ids_cache: Dict[str: Tuple[str]] = {}

    col = 1
    while len(current_col_elems_ids):
      next_col_elems_ids: List[str] = []
      for _id in current_col_elems_ids:
        if 'col' not in structure[_id]:

          target_nodes_ids_cache[_id] = target_nodes_ids = \
            tuple(self._get_connected_nodes_ids(_id, structure, 'target'))

          source_nodes_ids_cache[_id] = source_nodes_ids = \
            tuple(self._get_connected_nodes_ids(_id, structure, 'source'))

          if len(source_nodes_ids) < 2:

            structure[_id]['col'] = col

            for targ_node_id in target_nodes_ids:
              next_col_elems_ids.append(targ_node_id)

          else:
            delayed_processing_queue.add(_id)

      ids_to_remove_from_queue = []
      for _id in delayed_processing_queue:
        checked_reasons: List[bool] = list(map(
          lambda x: 'col' in structure[x],
          source_nodes_ids_cache[_id]))

        are_source_nodes_placed_in_this_col = reduce(
          lambda x, y: ('col' in structure[x] and structure[x]['col'] == col)
          or ('col' in structure[y] and structure[y]['col'] == col),
          source_nodes_ids_cache[_id])

        is_element_needs_handling = reduce(
          lambda x, y: x and y, checked_reasons
        ) and not are_source_nodes_placed_in_this_col

        if is_element_needs_handling:
          ids_to_remove_from_queue.append(_id)

          if 'col' not in structure[_id]:

            structure[_id]['col'] = col

            for targ_node_id in target_nodes_ids_cache[_id]:
              next_col_elems_ids.append(targ_node_id)

      while len(ids_to_remove_from_queue):
        _id = ids_to_remove_from_queue.pop()
        delayed_processing_queue.remove(_id)

      current_col_elems_ids = next_col_elems_ids
      col += 1

  def calc_grid_sizes(self):
    """
    Рассчитываем положение каждого элемента в сетке исходя из его
    ветки, номера в ветке и дорожки. Затем рассчитываем структуру
    self.grid в формате:
    {"cols": [0, 0, 0, ..], "rows": [0, 0, 0, ..]}
    """
    filtered_process = list(filter(
      lambda x: x['tag'] not in ['sequenceFlow', 'laneSet'],
      self.repr.values()))
    self.num_of_brunches = reduce(
      lambda acc, item: max(acc, item['branch']), filtered_process, 0)

    for subprocess in reversed(self.subprocesses):
      params_list = []
      for _id, elem in subprocess.items():

        if elem['tag'] in ['laneSet', 'sequenceFlow', 'outgoing', 'incoming']:
          continue

        # все элементы развернутого подпроцесса находятся в одной дорожке
        # (см ограничения в readme)
        params_list.append(
          self._calc_element_grid_params(elem, subprocess.lane))

      self._update_grid(subprocess.grid, params_list)

    params_list = []
    for _id, elem in self.repr.items():

      if elem['tag'] in ['laneSet', 'sequenceFlow', 'outgoing', 'incoming']:
        continue

      lane = self._get_elem_lane_number(_id)
      params_list.append(self._calc_element_grid_params(elem, lane))

    self._update_grid(self.grid, params_list)

  def _calc_element_grid_params(self, elem, lane):

    subprocess_width, subprocess_height = None, None
    if elem['tag'] == 'subProcess':
      subprocess = list(
        filter(lambda x: x.id == elem['id'], self.subprocesses))[0]
      subprocess_width, subprocess_height = \
        sum(subprocess.grid['cols']), sum(subprocess.grid['rows'])

    return {
      'id': elem['id'],
      'c': (elem['col'] - 1) * self.num_of_brunches + elem['branch'],
      'r': (lane - 1) * self.num_of_brunches + elem['branch'],
      'w': subprocess_width or self.sizes[elem['id']]['width'],
      'h': subprocess_height or self.sizes[elem['id']]['height']}

  def _update_grid(self, grid, params_list):
    self.elem_params.update({i['id']: i for i in params_list})

    max_col_idx = max(map(lambda x: x['c'], params_list))
    max_row_idx = max(map(lambda x: x['r'], params_list))

    grid['cols'] = [0.0] * max_col_idx
    grid['rows'] = [0.0] * max_row_idx

    # todo выделить в одну функцию для строк и колонок
    for idx, _ in enumerate(grid['cols']):
      try:
        grid['cols'][idx] = max(map(
          lambda x: x['w'],
          list(filter(lambda y: y['c'] == idx + 1, params_list))
        )) + 2 * self.visual_indent
      except ValueError:
        grid['cols'][idx] = 0.0

    for idx, _ in enumerate(grid['rows']):
      try:
        grid['rows'][idx] = max(map(
          lambda x: x['h'],
          list(filter(lambda y: y['r'] == idx + 1, params_list))
        )) + 2 * self.visual_indent
      except ValueError:
        grid['rows'][idx] = 0.0

  def _get_elem_lane_number(self, elem_id):
    lane_set_id = None
    for key in self.repr.keys():
      if self.repr[key]['tag'] == 'laneSet':
        lane_set_id = key
        break

    lanes = list(self.repr[lane_set_id]["children"].values())
    for idx, lane in enumerate(lanes):
      if elem_id in lane['children']:
        return idx + 1
    return None

  def calc_elems_coords(self, structure):
    """размещаем элементы внутри ячеек сетки (считаем координаты)"""

    subprocess_shift_left, subprocess_shift_top = 0.0, 0.0
    try:
      grid = structure.grid
      subprocess_shift_left = self.elem_params[structure.id]['x']
      subprocess_shift_top = self.elem_params[structure.id]['y']
    except AttributeError:
      grid = self.grid

    for key in structure.keys():

      if key not in self.elem_params:
        continue

      params = self.elem_params[key]

      cell_width = grid['cols'][params['c'] - 1]
      cell_height = grid['rows'][params['r'] - 1]

      accumulated_width = sum(grid['cols'][:params['c'] - 1])
      accumulated_height = sum(grid['rows'][:params['r'] - 1])

      params.update({
        'x': accumulated_width +
             (cell_width - params['w']) / 2 +
             subprocess_shift_left,
        'y': accumulated_height +
             (cell_height - params['h']) / 2 +
             subprocess_shift_top})

  def update_pool(self):
    for elem in self.repr.values():
      if elem['tag'] != 'laneSet':
        continue

      lanes_count = len(elem['children'])
      lane_size = len(self.grid['rows']) // lanes_count
      lanes = [self.grid['rows'][i * lane_size:(i + 1) * lane_size] for i in
               range(lanes_count)]
      heights = [sum(lane) for lane in lanes]
      width = sum(self.grid['cols'])

      self.elem_params['laneSet'] = {
        'x': 0.0,
        'y': 0.0,
        'w': width,
        'h': sum(heights),
        'spec': 'laneSet'}

      y_accumulator = 0
      for i, lane in enumerate(elem['children'].values()):
        self.elem_params[lane['id']] = {
          'x': 0.0,
          'y': y_accumulator,
          'w': width,
          'h': heights[i],
          'spec': 'lane'}
        y_accumulator += heights[i]

      break

  def calc_edges(self):
    """считаем координаты стрелок"""
    pass


class BpmnDiEditorGui(tk.Tk):
  def __init__(self):
    super().__init__()
    self.title('BPMN Viewer')

    self.entry_file = None
    self.chk_var = None
    self.text_xml = None

    self.bg_color = '#2B2B2B'
    self.text_bg_color = '#1E1E1E'
    self.button_bg_color = '#3D59AB'
    self.button_fg_color = '#CCCCCC'
    self.scrollbar_color = '#333333'
    self.toolbar_bg_color = '#2B2B2B'
    self.border_color = '#444444'
    self.checkbox_bg_color = '#2B2B2B'
    self.checkbox_fg_color = '#BBBBBB'
    self.checkbox_ind_color = '#FFBBBB'

    self.configure(bg=self.bg_color)
    style = ttk.Style()
    style.theme_use('clam')
    style.map('TCheckbutton', background=[('', self.checkbox_bg_color)],
              foreground=[('', self.checkbox_fg_color)],
              indicatorcolor=[('', self.checkbox_ind_color)])
    style.map('TScrollbar', background=[('', self.scrollbar_color)],
              troughcolor=[('', self.scrollbar_color)],
              darkcolor=[('', self.scrollbar_color)],
              lightcolor=[('', self.scrollbar_color)],
              sliderrelief=[('', 'flat')])

    self.create_top_panel()
    self.create_text_area()

    self.processor = BpmnXmlManager()
    self.layout_generator = BpmnLayoutGenerator()
    self.highlighter = XMLHighlighter(self.text_xml)

  def create_top_panel(self):
    toolbar_frame = tk.Frame(self, bg=self.toolbar_bg_color)
    toolbar_frame.pack(fill=tk.X, side=tk.TOP, padx=10, pady=10)

    self.entry_file = tk.Entry(toolbar_frame, width=50,
                               fg=self.button_fg_color, bg=self.text_bg_color,
                               bd=2, relief=tk.SOLID,
                               insertbackground=self.button_fg_color,
                               highlightthickness=0,
                               highlightbackground=self.border_color)
    self.entry_file.pack(side=tk.LEFT)
    self.entry_file.insert(tk.END, '/diagram.bpmn')

    btn_select_file = tk.Button(toolbar_frame, text="Выбрать файл",
                                command=self.select_file,
                                bg=self.button_bg_color,
                                fg=self.button_fg_color,
                                activebackground=self.button_bg_color,
                                activeforeground=self.button_fg_color, bd=2,
                                relief=tk.RIDGE, highlightthickness=0,
                                highlightbackground=self.border_color)
    btn_select_file.pack(side=tk.LEFT, padx=5)

    btn_calculate_di = tk.Button(toolbar_frame, text="Пересчитать di слой",
                                 command=self.convert_xml,
                                 bg=self.button_bg_color,
                                 fg=self.button_fg_color,
                                 activebackground=self.button_bg_color,
                                 activeforeground=self.button_fg_color, bd=2,
                                 relief=tk.RIDGE, highlightthickness=0,
                                 highlightbackground=self.border_color)
    btn_calculate_di.pack(side=tk.LEFT, padx=5)

    btn_save_file = tk.Button(toolbar_frame, text="Сохранить файл",
                              command=self.save_file, bg=self.button_bg_color,
                              fg=self.button_fg_color,
                              activebackground=self.button_bg_color,
                              activeforeground=self.button_fg_color, bd=2,
                              relief=tk.RIDGE, highlightthickness=0,
                              highlightbackground=self.border_color)
    btn_save_file.pack(side=tk.LEFT, padx=5)

  def create_text_area(self):
    text_frame = tk.Frame(self, bg=self.bg_color)
    text_frame.pack(fill=tk.BOTH, expand=True)

    self.chk_var = tk.BooleanVar(value=True)
    check_button = ttk.Checkbutton(text_frame, text="Включить подсветку",
                                   variable=self.chk_var,
                                   command=self.toggle_highlight)
    check_button.pack(side=tk.BOTTOM, anchor=tk.E)

    scrollbar_y = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
    scrollbar_x = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL)
    self.text_xml = tk.Text(text_frame, height=20, width=80, wrap=tk.NONE,
                            yscrollcommand=scrollbar_y.set,
                            xscrollcommand=scrollbar_x.set,
                            fg=self.button_fg_color, bg=self.text_bg_color,
                            insertbackground=self.button_fg_color,
                            font=("Consolas", 12))
    scrollbar_y.config(command=self.text_xml.yview)
    scrollbar_x.config(command=self.text_xml.xview)
    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
    self.text_xml.pack(expand=True, fill=tk.BOTH)

  def toggle_highlight(self):
    self.highlighter.is_enabled = not self.highlighter.is_enabled
    if self.highlighter.is_enabled:
      self.chk_var.set(True)
      self.highlighter.highlight()
    else:
      self.chk_var.set(False)
      for tag_type in self.highlighter.tags.keys():
        self.text_xml.tag_remove(tag_type, "1.0", tk.END)

  def select_file(self):
    file_path = filedialog.askopenfilename(
      filetypes=[("BPMN files", "*.bpmn")])
    if file_path:
      self.entry_file.delete(0, tk.END)
      self.entry_file.insert(tk.END, file_path)
      try:
        with open(file_path, 'r') as f:
          self.text_xml.delete('1.0', tk.END)
          self.text_xml.insert(tk.END, f.read())
          self.highlighter.highlight()
      except Exception as e:
        print(f'Ошибка чтения файла {file_path}: {e}')

  def save_file(self):
    current_filename = self.entry_file.get().strip()
    base, ext = os.path.splitext(current_filename)
    new_filename = f'{base}-autolayout{ext}'
    try:
      with open(new_filename, 'w') as f:
        f.write(self.text_xml.get("1.0", tk.END))
      print(f'Файл успешно сохранён: {new_filename}')
    except Exception as e:
      print(f'Ошибка записи файла: {e}')

  def on_focus_out(self, event):
    self.highlighter.highlight()

  def convert_xml(self):
    input_xml = self.text_xml.get('1.0', tk.END).strip()
    try:
      self.processor.set_input_xml(input_xml)
      self.processor.collect_sizes()
      process_elems = self.processor.extract_process_dict_repr()

      # todo классы запутались: передавать процессор в генератор целиком ?
      self.layout_generator.generate_di_layer(
        process_elems, self.processor.sizes_dict)
      di_layer_xml = self.processor.generate_di_layer_xml(
        self.layout_generator.elem_params)

      self.text_xml.delete('1.0', tk.END)
      self.text_xml.insert(tk.END, di_layer_xml)

    except ElementTree.ParseError:
      self.text_xml.insert(tk.END, 'XML is not valid')


class XMLHighlighter:
  def __init__(self, text_widget):
    self.text_widget = text_widget
    self.is_enabled = True
    self.tags = {
      "open_tag": {"pattern": r'<[\w:]+(?=[^>]*(>|/>))', "color": "#FF6F00"},
      "close_tag": {"pattern": r'</[\w:]+>', "color": "#555555"},
      "special_char": {
          "pattern": r'[<>=/?":]|bpmn|xml|xmlns|bpmndi|dc|di',
          "color": "#555555"},
      "attr_name": {"pattern": r'\w+(?=\s*=)', "color": "#1DA1F2"},
      "attr_value": {"pattern": r'"[^"]+"', "color": "#7ddba3"}}
    self.setup_tags()

  def setup_tags(self):
    for tag_type, config in self.tags.items():
      self.text_widget.tag_configure(tag_type, foreground=config["color"])

  def convert_to_tkindex(self, position):
    widget = self.text_widget
    cursor_position = widget.index(f"1.0+{position}c")
    return cursor_position

  def highlight(self):
    if not self.is_enabled:
      return
    content = self.text_widget.get("1.0", tk.END)
    for tag_type, config in self.tags.items():
      matches = [(m.start(), m.end()) for m in
                 re.finditer(config['pattern'], content)]
      for start, end in matches:
        char_start = self.convert_to_tkindex(start)
        char_end = self.convert_to_tkindex(end)
        self.text_widget.tag_add(tag_type, char_start, char_end)


if __name__ == '__main__':
  app = BpmnDiEditorGui()
  app.mainloop()
