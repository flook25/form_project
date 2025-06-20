import "./Header.css";

export default function Header() {
  return (
    <header>
      <div className="logo">
        <span>Task Management</span>
      </div>
      <div className="theme-container">
        <span>Night Mode</span>
        <span className="icon">Changing Mode</span>
      </div>
    </header>
  );
}
