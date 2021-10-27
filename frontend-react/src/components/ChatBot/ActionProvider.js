// ActionProvider starter code
class ActionProvider {
    constructor(createChatBotMessage, setStateFunc, createClientMessage) {
      this.createChatBotMessage = createChatBotMessage;
      this.setState = setStateFunc;
      this.createClientMessage = createClientMessage;
    }

    helloWorldHandler = (message) => {
      // let message = "hi"
      // fetch("http://localhost:8000/chat/")
      // .then(resp => resp.json())
      // .then(message => {
      //   this.setChatbotMessage(this.createChatBotMessage(message.data))
      // })
      this.setChatbotMessage(this.createChatBotMessage(message))
    }

    setChatbotMessage = (message) => {
      this.setState(state => ({ ...state, messages: [...state.messages, message] }))
    }
  }

export default ActionProvider;