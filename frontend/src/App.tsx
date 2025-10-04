// App.tsx
import React from 'react';
import { HashRouter as Router, Route, Routes } from "react-router-dom";
import './App.css';
import Header from './component/mains/header/header'
import { MsalAuthenticationResult, MsalAuthenticationTemplate } from '@azure/msal-react';
import { InteractionType } from '@azure/msal-browser';
import Welcome from 'component/pages/welcome/welcome';
import Footer from 'component/mains/footer/footer';
import Cards from 'component/mains/card/cards';

const App: React.FC = () => {
  const LoadingComponent = () => (
    <div className="loading">
      <div className="loader"></div>
      <div className="loadingMessage">Loading in progress...</div>
    </div>
  );

  const ErrorComponent = ({error}: MsalAuthenticationResult) => (
    <p className="errorMessage">An Error Occurred: {error?.message}</p>
  );
  
  return (
    <MsalAuthenticationTemplate 
      interactionType={InteractionType.Redirect}
      errorComponent={ErrorComponent}
      loadingComponent={LoadingComponent}
    >   
      <Router>   
        <div className="App">
          <Routes> 
            <Route path="/" element={ <Welcome /> } />
            {/* <Route path="/" element={ <Cards /> } /> */}
          </Routes> 
        </div>
      </Router>
    </MsalAuthenticationTemplate>  
  );
}

export default App;
