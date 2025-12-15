import xml.etree.ElementTree as ElementTree
from tkinter import Tk, Label, Text, Button, END
import json
from typing import List


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
    self.root = ElementTree.ElementTree(ElementTree.fromstring(self.input_xml)).getroot()

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
    processes = root.findall(f'./{self.root_tag_name}', namespaces=self.namespaces)

    if not processes:
      raise ValueError("BPMN XML is not valid: aint no <bpmn:process> tag")

    first_process = processes[0]  # todo добавить поддержку схем с несколькими процессами
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

    self.add_structure_attrs()
    self.calc_grid_structure()
    self.calc_elem_sizes()
    self.calc_grid_sizes()
    self.calc_elem_coords()
    self.calc_edges()

  @staticmethod
  def _get_arrow_endpoint_node(arrow_id, side, structure):
    """mode: enum = 'source' | 'target' """
    return structure[arrow_id][f'{side}Ref']

  def _handle_target_nodes(self, source_node_id, structure):
    """
    Ищем следующие элементы. Первый возвращшаем, остальные помечаем к посещению.
    """
    outgoing_flows_keyvalues = filter(
      lambda x: x[1]['tag'] == 'outgoing', structure[source_node_id]['children'].items())
    target_nodes_ids = list(map(
      lambda x: self._get_arrow_endpoint_node(x[0], 'target', structure), outgoing_flows_keyvalues))

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
    next_elem_id = self._handle_target_nodes(initial_elem_id, structure)

    while next_elem_id:
      if structure[next_elem_id]['tag'] == 'subProcess':
        self.subprocesses.append(structure[next_elem_id]['children'])

      next_elem_id = self._handle_target_nodes(next_elem_id, structure)

    self.brunch_counter += 1

    next_brunch_first_elem = self.nodes_to_visit_ids.pop()
    try:
      self._traverse_and_assign_branch_numbers(next_brunch_first_elem, structure)
    except IndexError:
      pass

  def _handle_process(self, structure=None):
    """
    Собираем все стартовые события и запускаем разметку ветвей схемы.
    """
    self.brunch_counter = 1
    self.start_events_ids = []

    if not structure:
      structure = self.repr

    self.start_events_ids += [
      v['id'] for k, v in structure.items() if v.get('tag') == 'startEvent']

    for initial_elem_id in self.start_events_ids:
      self._traverse_and_assign_branch_numbers(initial_elem_id, structure)

  def add_structure_attrs(self):
    self._handle_process()
    while self.subprocesses:
      self._handle_process(self.subprocesses.pop())

  def calc_grid_structure(self):
    """считаем размерность сетки и адреса ячеек в ней.
        Должен вызываться рекурсивно для поддержки субпроцессов.
        Возможность наличия неограниченного количества субпроцессов
        учесть в размерах сетки и в принципах индексации"""
    pass

  def calc_elem_sizes(self):
    """считаем размер элементов"""
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
