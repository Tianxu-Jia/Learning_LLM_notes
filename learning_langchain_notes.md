## Building an application

The most common and most important chain that LangChain helps create contains three things:

-   LLM.
-   Prompt Templates
-   Output Parsers

## LLMs[](https://langchain-langchain.vercel.app/docs/get_started/quickstart#llms "Direct link to LLMs")

There are two types of language models:

-   LLMs: tring input, string output
-   ChatModels: list of `ChatMessage's` input, singal `ChatMessage` output

A `ChatMessage` has two required components:

-   `content`: This is the content of the message.
-   `role`: This is the role of the entity from which the `ChatMessage` is coming from.

LangChain provides several roles of `message`:

-   `HumanMessage`: A `ChatMessage` coming from a human/user.
-   `AIMessage`: A `ChatMessage` coming from an AI/assistant.
-   `SystemMessage`: A `ChatMessage` coming from the system.
-   `FunctionMessage`: A `ChatMessage` coming from a function call.
-   `ChatMessage`: If none of those roles sound right, there is also a `ChatMessage` class where you can specify the role manually.

LangChain provides a standard interface for `LLMs` and `ChatModels`:

-   `predict`: `Both LLMs and ChatModels` can takes in a string, returns a string
-   `predict_messages`: `Both LLMs and ChatModels` can  takes in a list of messages, returns a message.

Let's see how to work with these different types of models and their inputs. First, let's import an LLM and a ChatModel.

```
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

llm = OpenAI()
chat_model = ChatOpenAI()

llm.predict("hi!")
>>> "Hi"

chat_model.predict("hi!")
>>> "Hi"
```

The `OpenAI` and `ChatOpenAI` objects are basically just configuration objects. You can initialize them with parameters like `temperature` and others, and pass them around.

Next, let's use the `predict` method to run over a string input. This input is prompt.

```
text = "What would be a good company name for a company that makes colorful socks?"

llm.predict(text)
# >> Feetful of Fun

chat_model.predict(text)
# >> Socks O'Color
```

Finally, let's use the `predict_messages` method to run over a list of messages.

```
from langchain.schema import HumanMessage

text = "What would be a good company name for a company that makes colorful socks?"
messages = [HumanMessage(content=text)]

llm.predict_messages(messages) # `temperature can be configured here`
# >> Feetful of Fun

chat_model.predict_messages(messages) # `temperature can be configured here`
# >> Socks O'Color
```

For both these methods, you can also pass in parameters as key word arguments. For example, you could pass in `temperature=0` to adjust the temperature that is used from what the object was configured with. Whatever values are passed in during run time will always override what the object was configured with.

## Prompt templates[](https://langchain-langchain.vercel.app/docs/get_started/quickstart#prompt-templates "Direct link to Prompt templates")

Actually, prompts can be regarded as input to the LLM/ChatModel if we take them as functions in programming. In bussiness domain, we can think the LLMs/ChatModels as an assistant, the prompt is the instructions to them. They will do the the task according to them. In the previous example, the text we passed to the model contained instructions to generate a company name. 

`PromptTemplates` help with exactly this! They bundle up all the logic for going from user input into a fully formatted prompt. This can start off very simple - for example, a prompt to produce the above string would just be:

```
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")

prompt.format(product="colorful socks")

# from string template===>formated prompt string
```

```
What is a good name for a company that makes colorful socks?
```

ChatPromptTemplate can generate mssage list from f-string list.
```
from langchain.prompts.chat import ChatPromptTemplate

template = "You are a helpful assistant that translates {input_language} to {output_language}."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([    
    ("system", template),    
    ("human", human_template),])

chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")
```

```
[    
    SystemMessage(content="You are a helpful assistant that translates English to French.", additional_kwargs={}),    
    HumanMessage(content="I love programming.")
]
```

ChatPromptTemplates can also be constructed in other ways - see the [section on prompts](https://langchain-langchain.vercel.app/docs/modules/model_io/prompts) for more detail.

## Output parsers[](https://langchain-langchain.vercel.app/docs/get_started/quickstart#output-parsers "Direct link to Output parsers")

OutputParsers convert the raw output of an LLM into a format that can be used downstream. There are few main type of OutputParsers, including:

-   text ==> structured information (e.g. JSON)
-   ChatMessage ==> a string
-   Convert the extra information returned from a call besides the message (like OpenAI function invocation) into a string.

One example that converts a comma separated list into a list.

```
from langchain.schema import BaseOutputParser

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""
    def parse(self, text: str):
        """Parse the output of an LLM call."""        
        return text.strip().split(", ")
        
CommaSeparatedListOutputParser().parse("hi, bye")
# >> ['hi', 'bye']
```

## PromptTemplate + LLM + OutputParser[](https://langchain-langchain.vercel.app/docs/get_started/quickstart#prompttemplate--llm--outputparser "Direct link to PromptTemplate + LLM + OutputParser")

We can now combine all these into one chain. This chain works with the pipeline:

input variables ==> `prompt template` ==>prompt ==>`language model`==>`(optional) output parser`==>output. This is a convenient way to bundle up a modular piece of logic. Let's see it in action!

```
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""    
    
    def parse(self, text: str):        
        """Parse the output of an LLM call."""        
        return text.strip().split(", ")
    
template = """You are a helpful assistant who generates comma separated lists.A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.ONLY return a comma separated list, and nothing more."""
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([    
    ("system", template),    
    ("human", human_template),])

chain = chat_prompt | ChatOpenAI() | CommaSeparatedListOutputParser()

chain.invoke({"text": "colors"})

# >> ['red', 'blue', 'green', 'yellow', 'orange']
```

Note that we are using the `|` syntax to join these components together. This `|` syntax is called the LangChain Expression Language. 

# Model and its I/O

The model is the core element of any language application, and prompts and Output parsers are the interface to the model.

-   [Prompts](https://langchain-langchain.vercel.app/docs/modules/model_io/prompts/): Templatize, dynamically select, and manage model inputs
-   [Language models](https://langchain-langchain.vercel.app/docs/modules/model_io/models/): Make calls to language models through common interfaces
-   [Output parsers](https://langchain-langchain.vercel.app/docs/modules/model_io/output_parsers/): Extract information from model outputs

![model_io_diagram](https://langchain-langchain.vercel.app/assets/images/model_io-1f23a36233d7731e93576d6885da2750.jpg)

[

](https://langchain-langchain.vercel.app/docs/modules/)

# Prompts

A prompt for a language model is a set of instructions or input provided by a user to guide the model's response, helping it understand the context and generate relevant and coherent language-based output, such as answering questions, completing sentences, or engaging in a conversation.

# Prompt templates

A template may include instructions, few-shot examples, and specific context and questions appropriate for a given task.

LangChain creates model agnostic templates.

Typically, language models expect the prompt to either be a string or else a list of chat messages.

## Prompt template[](https://langchain-langchain.vercel.app/docs/modules/model_io//#prompt-template "Direct link to Prompt template")

Use `PromptTemplate` to create a template for a string prompt.

By default, `PromptTemplate` uses [Python's str.format](https://docs.python.org/3/library/stdtypes.html#str.format) syntax for templating; however other templating syntax is available (e.g., `jinja2`).

```
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(    
    "Tell me a {adjective} joke about {content}."
    )
prompt_template.format(adjective="funny", content="chickens")
```

```
"Tell me a funny joke about chickens."
```

The template supports any number of variables, including no variables:

```
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Tell me a joke"
    )
prompt_template.format()
```

For additional validation, specify `input_variables` explicitly. These variables will be compared against the variables present in the template string during instantiation, raising an exception if there is a mismatch; for example(This seems for validation only!?),

```
from langchain.prompts import PromptTemplate

invalid_prompt = PromptTemplate(    
    input_variables=["adjective"],    
    template="Tell me a {adjective} joke about {content}."
)
```

## Chat prompt template[](https://langchain-langchain.vercel.app/docs/modules/model_io//#chat-prompt-template "Direct link to Chat prompt template")

The prompt to [chat models](https://langchain-langchain.vercel.app/docs/modules/model_io/prompts/models/chat) is a list of chat messages.

Each chat message is associated with `content`
and `role`. 

Create a chat prompt template:

```
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([    
    ("system", "You are a helpful AI bot. Your name is {name}."),    
    ("human", "Hello, how are you doing?"),    
    ("ai", "I'm doing well, thanks!"),    
    ("human", "{user_input}"),])

messages = template.format_messages(    
    name="Bob",    
    user_input="What is your name?"
)
```

`ChatPromptTemplate.from_messages` accepts a variety of message representations.

For example, in addition to using the 2-tuple representation of (type, content) used above, you could pass in an instance of `MessagePromptTemplate` or `BaseMessage`.

```
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate

template = ChatPromptTemplate.from_messages(    
    [        
        SystemMessage(            
            content=(                
                "You are a helpful assistant that re-writes the user's text to sound more upbeat."            
            )        
        ),        
        HumanMessagePromptTemplate.from_template("{text}"  # return an object as HumanMessage
        ),    
    ]
)

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI()
llm(template.format_messages(text='i dont like eating tasty things.'))
```

```
AIMessage(content='I absolutely adore indulging in delicious treats!', additional_kwargs={}, example=False)
```

This provides you with a lot of flexibility in how you construct your chat prompts.

# Connecting to a Feature Store

# Custom prompt template

Let's suppose we want the LLM to generate English language explanations of a function given its name. To achieve this task, we will create a custom prompt template that takes in the function name as input, and formats the prompt template to provide the source code of the function.

## Why are custom prompt templates needed?[](https://langchain-langchain.vercel.app/docs/modules/model_io//#why-are-custom-prompt-templates-needed "Direct link to Why are custom prompt templates needed?")

LangChain provides a set of [default prompt templates](https://langchain-langchain.vercel.app/docs/modules/model_io/prompts/prompt_templates/) that can be used to generate prompts for a variety of tasks. However, there may be cases where the default prompt templates do not meet your needs. For example, you may want to create a prompt template with specific dynamic instructions for your language model. In such cases, you can create a custom prompt template.

## Creating a custom prompt template[](https://langchain-langchain.vercel.app/docs/modules/model_io//#creating-a-custom-prompt-template "Direct link to Creating a custom prompt template")

There are essentially two distinct prompt templates available - string prompt templates and chat prompt templates. String prompt templates provides a simple prompt in string format, while chat prompt templates produces a more structured prompt to be used with a chat API.

In this guide, we will create a custom prompt using a string prompt template.

To create a custom string prompt template, there are two requirements:

1.  It has an input\_variables attribute that exposes what input variables the prompt template expects.
2.  It defines a format method that takes in keyword arguments corresponding to the expected input\_variables and returns the formatted prompt.

We will create a custom prompt template that takes in the function name as input and formats the prompt to provide the source code of the function. To achieve this, let's first create a function that will return the source code of a function given its name.

```
import inspectdef get_source_code(function_name):    # Get the source code of the function    return inspect.getsource(function_name)
```

Next, we'll create a custom prompt template that takes in the function name as input, and formats the prompt template to provide the source code of the function.

```
from langchain.prompts import StringPromptTemplatefrom pydantic import BaseModel, validatorPROMPT = """\Given the function name and source code, generate an English language explanation of the function.Function Name: {function_name}Source Code:{source_code}Explanation:"""class FunctionExplainerPromptTemplate(StringPromptTemplate, BaseModel):    """A custom prompt template that takes in the function name as input, and formats the prompt template to provide the source code of the function."""    @validator("input_variables")    def validate_input_variables(cls, v):        """Validate that the input variables are correct."""        if len(v) != 1 or "function_name" not in v:            raise ValueError("function_name must be the only input_variable.")        return v    def format(self, **kwargs) -> str:        # Get the source code of the function        source_code = get_source_code(kwargs["function_name"])        # Generate the prompt to be sent to the language model        prompt = PROMPT.format(            function_name=kwargs["function_name"].__name__, source_code=source_code        )        return prompt    def _prompt_type(self):        return "function-explainer"
```

#### API Reference:

-   [StringPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain.prompts.base.StringPromptTemplate.html)

## Use the custom prompt template[](https://langchain-langchain.vercel.app/docs/modules/model_io//#use-the-custom-prompt-template "Direct link to Use the custom prompt template")

Now that we have created a custom prompt template, we can use it to generate prompts for our task.

```
fn_explainer = FunctionExplainerPromptTemplate(input_variables=["function_name"])# Generate a prompt for the function "get_source_code"prompt = fn_explainer.format(function_name=get_source_code)print(prompt)
```

```
    Given the function name and source code, generate an English language explanation of the function.    Function Name: get_source_code    Source Code:    def get_source_code(function_name):        # Get the source code of the function        return inspect.getsource(function_name)        Explanation:    
```

# Few-shot prompt templates

A few-shot prompt template can be constructed from either a set of examples, or from an Example Selector object.

### Use Case[](https://langchain-langchain.vercel.app/docs/modules/model_io//#use-case "Direct link to Use Case")

In this tutorial, we'll configure few-shot examples for self-ask with search.
