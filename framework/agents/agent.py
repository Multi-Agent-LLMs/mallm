import xml.etree.ElementTree as ET
import glob, os

class Agent():
    def __init__(self, id, llm_model, persona):
        self.id = id
        self.llm = llm_model
        self.persona = persona
        import xml.etree.ElementTree as ET
        self.memory_bucket= "../../experiments/memory_bucket/{}.xml".format(self.id)
        root = ET.Element("root")
        tree = ET.ElementTree(root)
        memory = ET.SubElement(root, 'memory')
        tree.write(self.memory_bucket)
        

    def createDraft():
        return None

    def updateMemory(self, agent_id, text):
        tree = ET.parse(self.memory_bucket)
        root = tree.getroot()
        memory = root.find("memory")
        
        # Append the new text to the existing memory
        text_element = ET.SubElement(memory, 'text')
        text_element.set('agent_id', str(agent_id))
        text_element.text = text
        
        tree.write(self.memory_bucket)


filelist = glob.glob(os.path.join("../../experiments/memory_bucket/", "*.xml"))
for f in filelist:
    os.remove(f)

agent = Agent(1, "model placeholder", "Health Minister")
agent.updateMemory(1, "I like tea.")
agent.updateMemory(2, "I like milk.")
agent.updateMemory(1, "Tea and milk is nice.")
agent.updateMemory(2, "R u bri'ish?")