import "./Item.css";
import { IoTrash } from "react-icons/io5";

export default function Item(props) {
  const { data, deleteTask, seq, editTask } = props;
  return (
    <div className="list-item">
      <p className="title">
        {seq}. {data.title}
      </p>
      <div className="button-container">
        <button className="delete" onClick={() => deleteTask(data.id)}>
          <IoTrash size={20} />
        </button>
        <button className="edit" onClick={() => editTask(data.id)}>
          Edit
        </button>
      </div>
    </div>
  );
}