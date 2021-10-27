import React, { Component } from 'react'
import {BrowserRouter as Router, Route, Switch } from 'react-router-dom'
import './App.css';
import 'tachyons'
import SearchBox from './components/Search/SearchBox'
import CardList from './components/Card/CardList'
import DogInfo from './components/DogInfo/DogInfo'
import ImageSearch from './components/ImageSearch/ImageSearch'
import ChatBot from './components/ChatBot/ChatBot'
import { Link } from 'react-router-dom'
import Particles from 'react-particles-js';
import Logo from './components/Logo/Logo';


// import { dogs } from './dogs'

const particlesOptions = {
  particles: {
    number: {
      value: 30,
      density: {
        enable: true,
        value_area: 800
      }
    }
  }
}

class App extends Component {
  constructor() {
    super();
    this.state = {
      dogs: [],
      searchField: '',
      total_dogs: 30
    }
  }

  // componentDidMount() {
  //   fetch(`http://localhost:8000/dogs/${this.state.total_dogs}`)
  //   .then(response => response.json())
  //   .then(results => {
  //     this.setState({ dogs: results })
  //   })
  // }

  componentDidMount() {
    const HOSTNAME_TAG = "<hostname>"
    let BASE_API_URL = process.env.REACT_APP_BASE_API_URL;
    if(BASE_API_URL.includes(HOSTNAME_TAG)){
    //window.location.hostname
    BASE_API_URL = BASE_API_URL.replace(HOSTNAME_TAG, window.location.hostname);
  }

  fetch(`${BASE_API_URL}/dogs/${this.state.total_dogs}`)
  .then(response => response.json())
  .then(results => {
  this.setState({ dogs: results })
  })
}

  onSearchChange = (event) => {
    this.setState({ searchField: event.target.value })
  }

  onClick = () => {
    this.setState({ total_dogs: this.state.total_dogs + 20})
    this.componentDidMount()
  }

  render() {
    const filterDogs = this.state.dogs.filter(dog => {
      return dog.tags.toLowerCase().includes(this.state.searchField.toLowerCase())
    })

    return (
      <Router>
        <Logo />
        <Switch>
          <div className="tc ma5">
            <Particles className='particles'
            params={particlesOptions}
            />
            <Route exact path="/">
              <SearchBox searchChange={this.onSearchChange} />
              <Link to="/image-search">
                <button class="image-search-btn">Image Search</button>
              </Link>
              <CardList dogs={filterDogs} />
              <button onClick={this.onClick}>Load More</button>
            </Route>
            <Route path="/dog-info/:id" component={DogInfo} />
            <Route path="/image-search" component={ImageSearch} />
            <Route exact path="/chat">
              <ChatBot />
            </Route>
          </div>
        </Switch>
      </Router>
    )
  }
}

export default App;
