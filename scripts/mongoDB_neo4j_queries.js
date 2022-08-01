db.triples.aggregate([
    {$group: {_id: "$entity1", count:{$sum:1}}},
    {$sort:{"count":1}}
],
    {allowDiskUse:true}
);

db.triples.aggregate([
    {$group: {_id: "$entity1", count:{$sum:1}}},
    {$sort:{"count":1}},
    {$out: "entity1_triples"}
],
    {allowDiskUse:true}
);

db.triples.aggregate([
    {$group: {_id: "$entity2", count:{$sum:1}}},
    {$sort:{"count":1}},
    {$out: "entity2_triples"}
],
    {allowDiskUse:true}
);

db.entity1_triples.aggregate([
    {
        $lookup:
          {
            from: entity2_triples,
            localField: _id,
            foreignField: _id,
            as: <output array field>
          }
    }
],
    {allowDiskUse:true}
);

/*
- Sort by count Ascending
- Group by entity1 and entity2 and then add the counts together
*/

db.entity1_triples.aggregate([
    {$lookup: {
        from: "entity2_triples",
        localField: "_id",
        foreignField: "_id",
        as: "a"
    }},
    {$unwind: {path: "$a", preserveNullAndEmptyArrays: true}},
    {$project: {
        value: {$add: ["$count", {$ifNull: ["$a.count", 0]}]}
    }},
    {$out: "entity_counts"}
]);

// Sort by count and limit
db.entity_counts.sort({ "value": 1 }).limit(100)

// left-join on entity_counts
db.entity_counts.aggregate([
    {
        $lookup:
          {
            from: "entity_subset",
            localField: "_id",
            foreignField: "id",
            as: ""
          }
    }
],
    {allowDiskUse:true}
);

db.entity_subset.aggregate( [
   {
      $lookup: {
         from: "entity_counts",
         localField: "id",
         foreignField: "_id",
         as: "fromCounts"
      }
   },
   {
      $replaceRoot: { newRoot: { $mergeObjects: [ { $arrayElemAt: [ "$fromCounts", 0 ] }, "$$ROOT" ] } }
   },
   { $project: { fromCounts: 0 } },
   { $sort : { value : 1 } },
   { $out: "entityList" }
],
    {allowDiskUse:true}
);


// Duplicate Collection
pipeline = [ {"$match": {}},
             {"$out": "equivalence"},
]
db.properties.aggregate(pipeline)

// List of non-symmetric properties
pipeline = [
    {
        '$lookup':
          {
            'from': "symmetric",
            'localField': "_id",
            'foreignField': "r1.id",
            'as': "matched_relations"
          }
    },
    {
      '$match': {
        "matched_relations": { '$eq': [] }
      }
    },
    { '$out':'nsproperties' }
]

db.properties.aggregate(pipeline, allowDiskUse=True)

// Delete matched_relations
db.nsproperties.update_many({}, {'$unset':{"matched_relations":1}})


// NEO4J
CREATE CONSTRAINT UniqueEntity ON (e:Entity) ASSERT e.id IS UNIQUE

:auto USING PERIODIC COMMIT 1000
LOAD CSV FROM 'file:///triple_subset.txt' AS row FIELDTERMINATOR '\t'
MERGE (e1:Entity {id:row[0]})
MERGE (e2:Entity {id:row[2]})
MERGE (e1)-[rel:Relation {id: row[1]}]->(e2)
RETURN count(row);

:auto USING PERIODIC COMMIT 300
LOAD CSV FROM 'file:///triple_subset.txt' AS row FIELDTERMINATOR '\t'
MERGE (e1:Entity {id:row[0]})
MERGE (e2:Entity {id:row[2]});

:auto USING PERIODIC COMMIT 300
LOAD CSV FROM 'file:///triple_subset.txt' AS row FIELDTERMINATOR '\t'
MATCH (a:Entity), (b:Entity)
WHERE a.id = row[0] AND b.id = row[2]
MERGE (a)-[:Relation {id: row[1]}]->(b)
RETURN count(row);

// Symmetric Query
MATCH (e1:Entity)-[r1:Relation]->(e2:Entity)-[r2:Relation]->(e1:Entity)
WHERE r1.id = r2.id
RETURN DISTINCT r1.id

// Equivalence Query (NOT WORKING)
MATCH (e2:Entity)<-[r2:Relation]-(e1:Entity)-[r1:Relation]->(e2:Entity)
WHERE r2.id <> r1.id and r1.id < r2.id
RETURN DISTINCT r1.id, r2.id

// Want Inner Join!?
SELECT s.Relation
FROM triples s
JOIN triples r
  ON s.Entity1 = r.Entity2
  AND s.Entity2 = r.Entity1
  AND s.Relation = r.Relation
