import Header from "./components/Header";
import AddForm from "./components/AddForm";
import Item from "./components/Item";
import { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [tasks, setTasks] = useState(JSON.parse(localStorage.getItem('tasks')) || []);
  const [title, setTitle] = useState("");
  const [editId, setEditId] = useState(null);
  const [theme,setTheme] = useState('dark')
  useEffect(() => {
    localStorage.setItem('tasks',JSON.stringify(tasks))
  }, [tasks])

  const deleteTask = (id) => {
    setTasks(tasks.filter((task) => task.id !== id));
  };

  const editTask = (id) => {
    const task = tasks.find((task) => task.id === id);
    setEditId(id);
    setTitle(task.title);
  };
  const addTask = () => {
    if (title.trim() === "") return;
    if (editId !== null) {
      setTasks(
        tasks.map((task) => (task.id === editId ? { ...task, title } : task))
      );
      setEditId(null);
    } else {
      setTasks([...tasks, { id: Date.now(), title }]);
    }
    setTitle("");
  };

  return (
    <div className={"App "+theme}>
      <Header theme={theme} setTheme={setTheme}/>
      <div className="container">
        <AddForm
          title={title}
          setTitle={setTitle}
          onAddTask={addTask}
          editId={editId}
        />
        <section>
          {tasks.map((data, id) => (
            <Item
              key={data.id}
              data={data}
              deleteTask={deleteTask}
              editTask={editTask}
              seq={id + 1}
            />
          ))}
        </section>
      </div>
    </div>
  );
}

export default App;
