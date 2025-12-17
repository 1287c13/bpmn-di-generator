import xml.etree.ElementTree as ElementTree
from tkinter import Tk, Label, Text, Button, END
import json
from typing import List, Literal, Dict, Set, Tuple
from functools import reduce


class BpmnXmlManager:
  def __init__(self, input_xml_path='diagram.bpmn'):
    self.namespaces = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    self.root_tag_name = 'bpmn:process'
    self.input_xml_path = input_xml_path
    self.input_xml = None
    self.root = None

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
  def parse_element(element):
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

  @staticmethod
  def generate_di_layer_xml(di_layer_dict):
    return json.dumps(di_layer_dict, indent=2)


class BpmnLayoutGenerator:

  def __init__(self):
    self.brunch_counter: int = 1
    self.repr: dict = {}
    self.subprocesses: list = []
    self.start_events_ids: List[str] = []
    self.visited_nodes_ids: List[str] = []
    self.nodes_to_visit_ids: List[str] = []

  def generate_di_layer(self, tags_dict_repr):
    self.repr = list(tags_dict_repr.values())[0]['children']

    self.call_process_handler('_add_structure_attrs_for_process')
    self.call_process_handler('_calc_grid_structure_for_process')
    self.call_process_handler('_calc_elem_sizes_for_process')
    self.calc_grid_sizes()
    self.calc_elem_coords()
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
    Ищем следующие элементы. Первый возвращшаем,
    остальные помечаем к посещению.
    """
    target_nodes_ids = self._get_connected_nodes_ids(
      parent_node_id, structure, 'target')

    self.nodes_to_visit_ids += target_nodes_ids[1:]

    try:
      structure[target_nodes_ids[0]].setdefault('brunch', self.brunch_counter)
      return target_nodes_ids[0]
    except IndexError:
      return None

  def _traverse_and_assign_branch_numbers(self, initial_elem_id, structure):
    """
    Проходим всю ветку начального элемента и делаем рекурсивный вызов
    для следующих обнаруженных элементов, которые нужно посетить.
    """
    structure[initial_elem_id]['brunch'] = self.brunch_counter
    next_elem_id = self._explore_neighboring_nodes(initial_elem_id, structure)

    while next_elem_id:
      if structure[next_elem_id]['tag'] == 'subProcess':
        self.subprocesses.append(structure[next_elem_id]['children'])

      next_elem_id = self._explore_neighboring_nodes(next_elem_id, structure)

    self.brunch_counter += 1

    next_brunch_first_elem = self.nodes_to_visit_ids.pop()
    try:
      self._traverse_and_assign_branch_numbers(next_brunch_first_elem,
                                               structure)
    except IndexError:
      pass

  @staticmethod
  def _get_start_events_ids(process):
    return [v['id'] for k, v in process.items() if
            v.get('tag') == 'startEvent']

  def _add_structure_attrs_for_process(self, structure):
    """
    Собираем все стартовые события и запускаем разметку ветвей схемы.
    """
    self.brunch_counter = 1
    self.start_events_ids = self._get_start_events_ids(structure)
    for initial_elem_id in self.start_events_ids:
      self._traverse_and_assign_branch_numbers(initial_elem_id, structure)

  def call_process_handler(self, handler_name):
    self.subprocesses = []
    getattr(self, handler_name)(self.repr)
    while self.subprocesses:
      getattr(self, handler_name)(self.subprocesses.pop())

  def _calc_grid_structure_for_process(self, structure):
    """
    Итерируемся по номерам столбцов и подбираем элементы для размещения в них.
    todo refactor this: extract helpers at least
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
            if structure[_id]['tag'] == 'subProcess':
              self.subprocesses.append(structure[_id]['children'])

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
          lambda x, y: x and y, checked_reasons) \
                                    and not are_source_nodes_placed_in_this_col

        if is_element_needs_handling:
          ids_to_remove_from_queue.append(_id)

          if 'col' not in structure[_id]:

            structure[_id]['col'] = col
            if structure[_id]['tag'] == 'subProcess':
              self.subprocesses.append(structure[_id]['children'])

            for targ_node_id in target_nodes_ids_cache[_id]:
              next_col_elems_ids.append(targ_node_id)

      while len(ids_to_remove_from_queue):
        _id = ids_to_remove_from_queue.pop()
        delayed_processing_queue.remove(_id)

      current_col_elems_ids = next_col_elems_ids
      col += 1

  def _calc_elem_sizes_for_process(self, structure):
    pass

  def calc_grid_sizes(self):
    """считаем размер ячеек"""
    pass

  def calc_elem_coords(self):
    """размещаем элементы по сетке (считаем координаты)"""
    pass

  def calc_edges(self):
    """считаем координаты стрелок"""
    pass


class BpmnDiEditorGui(Tk):
  def __init__(self, processor, layout_generator):
    super().__init__()
    self.title('BPMN DI Editor')
    self.processor = processor
    self.layout_generator = layout_generator
    self.xml_input_field = None
    self.xml_output_field = None

    self._create_ui()

  def _create_ui(self):
    Label(self, text='Input XML:').grid(row=0, column=0, sticky='W')
    self._add_text_field('xml_input_field', 0)
    Button(self, text='Convert',
           command=self.convert_xml).grid(row=2, column=0, columnspan=2)
    Label(self, text='Output XML:').grid(row=3, column=0, sticky='W')
    self._add_text_field('xml_output_field', 3)

    self.xml_input_field.insert(END, self.processor.input_xml or '')

  def _add_text_field(self, field_name, row_number):
    setattr(self, field_name, Text(self, wrap='none'))
    getattr(self, field_name).grid(row=row_number, column=1, sticky='NSEW')

  def convert_xml(self):
    self.xml_output_field.delete('1.0', END)

    input_xml = self.xml_input_field.get('1.0', END).strip()
    try:
      self.processor.set_input_xml(input_xml)
      process_elems = self.processor.extract_process_dict_repr()
      self.layout_generator.generate_di_layer(process_elems)
      di_layer_xml = self.processor.generate_di_layer_xml(
        self.layout_generator.repr)

      self.xml_output_field.insert(END, di_layer_xml)

    except ElementTree.ParseError:
      self.xml_output_field.insert(END, 'XML is not valid')


if __name__ == '__main__':
  xml_processor = BpmnXmlManager()
  bpmn_layout_generator = BpmnLayoutGenerator()
  app = BpmnDiEditorGui(xml_processor, bpmn_layout_generator)
  app.mainloop()
