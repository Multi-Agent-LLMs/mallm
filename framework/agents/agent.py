import xml.etree.ElementTree as ET
import glob, os

class Agent():
    def __init__(self, id, llm_model, persona):
        self.id = id
        self.llm = llm_model
        self.persona = persona
        self.memory_bucket= "/beegfs/wahle/github/MALLM/experiments/memory_bucket/{}.xml".format(self.id)
        root = ET.Element("root")
        tree = ET.ElementTree(root)
        ET.SubElement(root, 'agent_id').text = str(self.id)
        ET.SubElement(root, 'model').text = "LLM"
        ET.SubElement(root, 'persona').text = self.persona
        memory = ET.SubElement(root, 'memory')
        tree.write(self.memory_bucket)

    def createDraft():
        return None

    def updateMemory(self, agent_id, turn, text):
        tree = ET.parse(self.memory_bucket)
        root = tree.getroot()
        memory = root.find("memory")
        
        # Append the new text to the existing memory
        text_element = ET.SubElement(memory, 'text')
        text_element.set('agent_id', str(agent_id))
        text_element.set('turn', str(turn))
        text_element.text = text
        
        tree.write(self.memory_bucket)


filelist = glob.glob(os.path.join("../../experiments/memory_bucket/", "*.xml"))
for f in filelist:
    os.remove(f)

agent = Agent(1, "Model Placeholder", "Health Minister")
agent.updateMemory(1, 0, "I like tea.")
agent.updateMemory(2, 1, "I like milk.")
agent.updateMemory(1, 2, "Tea and milk is nice.")
agent.updateMemory(2, 3, "R u bri'ish?")