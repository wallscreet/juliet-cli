from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional
import yaml
from datetime import datetime
from string import Template


@dataclass
class ModelInstructions:
    """
    Iso configuration dataclass for managing prompts and messaging to the iso.
    """
    name: str = None
    description: str = None
    llm_model: str = None
    system_message: str = None
    assistant_intro: str = None
    assistant_focus: str = None
    commands: dict = None
    prompt_script: str = None
    start_token: str = None
    end_token: str = None
    mem_start_token: str = None
    mem_end_token: str = None
    history_start_token: str = None
    history_end_token: str = None
    chat_start_token: str = None
    chat_end_token: str = None
    completions_url: str = None
    conversations_filepath: str = None

    def __init__(self, method: str, assistant_name: str = None) -> None:
        """
        Model instructions init takes a method param as ['create', 'load'] to determine if the instructions should be loaded from a yaml file or created from the CLI.
        """
        self.conversations_filepath = f"isos/{assistant_name.lower()}/conversations.yaml"
        if method == 'load':
            if assistant_name:
                self.load_from_yaml(assistant_name)
                print(f"\nLoaded instructions for {self.name}")
                #print(asdict(self))
            else:
                print("Error: No iso name provided.")    
        elif method == 'create':
            self.load_defaults_from_yaml()
            print("Creating new iso instructions...")
            customize = input("Would you like to customize the instructions? (y/n): ").strip()
            if customize == 'y':
                instructions = self.to_dict()
                for key, value in instructions.items():
                    new_value = input(f"\n{key} ({value}): Press enter to keep current value or enter a new one: ").strip()
                    if new_value:
                        setattr(self, key, new_value)
                print(asdict(self))
            else:
                print("Using default instructions. You can customize these later.")
                print(asdict(self))
            
            # Create the iso directories and populate them with the default files
            isos_dir = Path('isos/')
            templates_dir = Path('iso-template/')
            templates = [file for file in templates_dir.iterdir() if file.suffix in ['.md', '.yml', '.yaml', '.txt']]
            # this gives the ability to default specifically named blank files and types for template inclusion
            include_files = []  

            directories = [
                'fine-tuning'
            ]

            try:
                print('Checking for isos directory...')
                if not isos_dir.exists():
                    os.mkdir(isos_dir)
                    print('----------------------------------------')
                    print('Directory (isos) created')
                
                print('Cross-checking for existing agents...')
                target_iso_dir = Path(f'isos/{self.name.lower()}')
                if target_iso_dir.exists():
                    print('----------------------------------------')
                    print(f'Iso ({self.name}) already exists. Pleae choose another name.')
                    return None
                else:
                    print('Iso does not exist, creating...')
                    target_iso_dir.mkdir(parents=True, exist_ok=True)
                    print('----------------------------------------')
                    print(f'Iso Directory ({self.name}) created')

                for directory in directories:
                    Path(f'isos/{self.name.lower()}/{directory}').mkdir(parents=True, exist_ok=True)
                    print('----------------------------------------')
                    print(f'Iso Sub-Directory ({self.name}/{directory}) created')
                
                # Copy the project template files
                for template in templates:
                    shutil.copy(template, f"{isos_dir}/{self.name.lower()}/{template.name}")
                    print(f"Copied {template} to {isos_dir}/{self.name.lower()}/{template.name}")

                print('----------------------------------------')
                print("All template files copied to new iso")
                print('----------------------------------------')

                self.save_to_yaml()

            except Exception as e:
                print(e)
                return

    def to_dict(self) -> dict:
        """
        Export config class to a base dict

        :returns: Base dictionary for the config class.
        """
        return asdict(self)
    
    def print_model_instructions(self) -> None:
        """
        Print the config to the terminal.

        :returns: Prints a pre-defined config string to the terminal.
        """
        print(f"Iso Configuration:\n{self.to_dict()}")
    
    def to_prompt_script(self, 
                         user_request: str, 
                         facts_context: Optional[List[str]] = None, 
                         mem_context: Optional[List[str]] = None, 
                         knowledge_context: Optional[List[str]] = None, 
                         chat_history: Optional[List[str]] = None, 
                         workspace_contents: str = None
    ) -> List[Dict[str, str]]:
        """
        Export instructions class as a prompt template.
        """

        facts_context_str = "\n".join(facts_context) if facts_context else "No related Facts found"
        mem_context_str = "\n".join(mem_context) if mem_context else "No related memories"
        knowledge_context_str = "\n".join(knowledge_context) if knowledge_context else "No related knowledge"
        chat_history_str = "\n".join(chat_history) if chat_history else "No chat history"
        current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")  # Thursday, December 18, 2025 at 02:30 PM

        messages = [
            {"role": "system", "content": f"<system>>{self.system_message}</system>\n"},
            {"role": "assistant", "content": f"\n<assistant>{self.assistant_intro}</assistant>\n"},
            {"role": "user", "content": f"\n<focus>Your current focus should be: {self.assistant_focus}</focus>\n"},
            {"role": "system", "content": f"\n<timestamp>The current date and time is:\n{current_time}</timestamp>\n"},
            {"role": "system", "content": f"\n<workspace>Workspace directory contents:\n{workspace_contents}</workspace>\n"},
            {"role": "system", "content": f"\n<facts>Facts from your Facts Table:\n{facts_context_str}</facts>\n"},
            {"role": "system", "content": f"\n<memory>Request context from your memory:\n{mem_context_str}</memory>\n"},
            {"role": "system", "content": f"\n<knowledge>Request context from your knowledge base:\n{knowledge_context_str}</knowledge>\n"},
            {"role": "system", "content": f"\n<history>Conversation chat history:\n{chat_history_str}</history>\n"},
            {"role": "user", "content": f"\n<user>User Request: {user_request}</user>\n"},
            {"role": "assistant", "content": "<assistant>\n"},
        ]

        return messages

    def to_prompt_script_md(self,
                            user_request: str,
                            facts_context: Optional[List[str]] = None,
                            mem_context: Optional[List[str]] = None,
                            knowledge_context: Optional[List[str]] = None,
                            chat_history: Optional[List[str]] = None,
                            workspace_contents: Optional[List[str]] = None,
                            todos: Optional[List[str]] = None,
    )-> List[Dict[str, str]]:
        """
        Create Markdown formatted instructions for the model
        """

        facts_context_str = "\n".join(facts_context) if facts_context else "No related Facts found"
        mem_context_str = "\n".join(mem_context) if mem_context else "No related memories"
        knowledge_context_str = "\n".join(knowledge_context) if knowledge_context else "No related knowledge"
        chat_history_str = "\n".join(chat_history) if chat_history else "No chat history"
        todos_str = ",\n".join(todos) if todos else "No TODOs"
        workspace_contents_str = "\n".join(workspace_contents) if workspace_contents else "No content in your Workspace"
        current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")  # Thursday, December 18, 2025 at 02:30 PM
        
        with open("templates/prompt_template.md") as f:
            t = Template(f.read())

        content = t.substitute(
                system_message=self.system_message,
                assistant_intro=self.assistant_intro,
                assistant_focus=self.assistant_focus,
                current_time=current_time,
                todos=todos_str,
                workspace_contents=workspace_contents_str,
                facts_context=facts_context_str,
                mem_context=mem_context_str,
                knowledge_context=knowledge_context_str,
                chat_history=chat_history_str,
        )

        messages = [
                {"role": "system", "content": f"{content}"},
                {"role": "user", "content": f"{user_request}"},
                {"role": "assistant", "content": "<assistant>"},
        ]

        return messages

    def update_model_instructions(self) -> None:
        """
        Iterate through the config and update the values or keep current.
        """
        instructions = self.to_dict()
        for key, value in instructions.items():
            new_value = input(f"\n{key} ({value}): Press enter to keep current value or enter a new one: ").strip()
            if new_value:
                setattr(self, key, new_value)
        self.save_to_yaml()
    
    def load_defaults_from_yaml(self) -> None:
        """
        Load the iso instructions config from a yaml file.
        """
        model_instructions = Path(f"iso-template/instructions.yaml")
        if model_instructions.exists():
            with model_instructions.open('r') as file:
                instructions = yaml.safe_load(file)
                for key, value in instructions.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

    def load_from_yaml(self, assistant_name: str) -> None:
        """
        Load the agent instructions config from a yaml file.
        """
        model_instructions = Path(f"isos/{assistant_name.lower()}/instructions.yaml")
        if model_instructions.exists():
            with model_instructions.open('r') as file:
                instructions = yaml.safe_load(file)
                for key, value in instructions.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
    
    def save_to_yaml(self) -> None:
        """
        Save the agent config to a yaml file.

        :returns: Saves the iso config to a yaml file.
        """
        data = self.to_dict()
        with open(f"isos/{self.name.lower()}/instructions.yaml", "w") as f:
            yaml.safe_dump(data, f)
