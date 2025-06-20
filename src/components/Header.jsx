import "./Header.css";
import { CiSun } from "react-icons/ci";
import { IoMoonOutline } from "react-icons/io5";

export default function Header(props) {
  const {theme, setTheme} = props;
  function ToggleTheme(){
    if(theme === 'light'){
      setTheme('dark')
    } else {
      setTheme('light')
    }
  }


  return (
    <header>
      <div className="logo">
        <span>Task Management</span>
      </div>
      <div className="theme-container">
        <span>{theme === 'light' ? 'light mode' : 'dark mode'}</span>
        <span className="icon" onClick={ToggleTheme}>{theme==='light' ? <CiSun></CiSun> : <IoMoonOutline></IoMoonOutline>}</span>
          
      </div>
    </header>
  );
}
