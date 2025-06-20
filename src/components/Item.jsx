import "./Item.css";
import React from "react";

export default function Item(props) {
  const { data, deleteTask, seq, editTask } = props;
  return (
    <div className="list-item">
      <p className="title">
        {seq}. {data.title}
      </p>
      <div className="button-container">
        <button className="delete" onClick={() => deleteTask(data.id)}>Delete</button>
        <button className="edit"   onClick={() => editTask(data.id)}>Edit</button>
      </div>
    </div>
  );
}
