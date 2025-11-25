import xml.etree.ElementTree as ElementTree
from tkinter import Tk, Label, Text, Button, END
import json


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
    result = {
      'tag': element.tag.split('}')[-1],
      'text': '' if not element.text else element.text.strip()
    }

    if element.attrib:
      result.update(element.attrib)

    children = list(element)
    if len(children) > 0:
      child_elements = []
      for child in children:
        child_elements.append(BpmnXmlManager.parse_element(child))

      result['children'] = child_elements

    return result

  def extract_process_dict_repr(self):
    root = ElementTree.fromstring(self.input_xml)
    processes = root.findall(f'./{self.root_tag_name}', namespaces=self.namespaces)

    if not processes:
      raise ValueError("BPMN XML is not valid: aint no <bpmn:process> tag")

    first_process = processes[0]
    data = BpmnXmlManager.parse_element(first_process)

    return data


class BpmnLayoutGenerator:
  @staticmethod
  def generate_di_layer(tags_dict_repr):
    return tags_dict_repr


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
    Button(self, text='Convert', command=self.convert_xml).grid(row=2, column=0, columnspan=2)
    Label(self, text='Output XML:').grid(row=3, column=0, sticky='W')
    self._add_text_field('xml_output_field', 3)

    self.xml_input_field.insert(END, self.processor.input_xml)

  def _add_text_field(self, field_name, row_number):
    setattr(self, field_name, Text(self, wrap='none'))
    getattr(self, field_name).grid(row=row_number, column=1, sticky='NSEW')

  def convert_xml(self):
    self.xml_output_field.delete('1.0', END)

    input_xml = self.xml_input_field.get('1.0', END).strip()
    try:
      self.processor.set_input_xml(input_xml)
      process_elems = self.processor.extract_process_dict_repr()
      di_layer_dict = self.layout_generator.generate_di_layer(process_elems)

      self.xml_output_field.insert(END, json.dumps(di_layer_dict, indent=2))  # debug !!

    except ElementTree.ParseError:
      self.xml_output_field.insert(END, 'XML is not valid')


if __name__ == '__main__':
  xml_processor = BpmnXmlManager()
  bpmn_layout_generator = BpmnLayoutGenerator()
  app = BpmnDiEditorGui(xml_processor, bpmn_layout_generator)
  app.mainloop()
