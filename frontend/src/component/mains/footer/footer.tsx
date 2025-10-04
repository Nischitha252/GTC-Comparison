// Footer.tsx
import { CONTACT_US, MICROSOFT_FORMS_URL } from 'config';
import './footer.css';

const Footer = () => {
  return (
    <footer className='footer'>
      <div className="footer-links">
        <a href={MICROSOFT_FORMS_URL}  
          target="_blank" 
          rel="noopener noreferrer"> 
          Feedback
        </a>

        <a href={CONTACT_US}>
          Contact Us 
        </a>
      </div>
      <div className="copyRight">© 2024 ABB</div>
    </footer>
  );
}

export default Footer;

