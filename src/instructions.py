from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
import yaml


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

    def __init__(self, method: str, assistant_name: str = None) -> None:
        """
        Model instructions init takes a method param as ['create', 'load'] to determine if the instructions should be loaded from a yaml file or created from the CLI.
        """
        #self.conversations_filepath = f"isos/{assistant_name.lower()}/conversations.yaml"
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
                'fine-tuning',
                'users'
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
